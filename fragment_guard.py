"""
Fragment Guard v5 — safe buffering with continuation-aware merge.

Design principles (incorporating reviewer feedback on v3 and v4):
  - NEVER delete messages from the shared LLMContext
  - NEVER mutate existing context messages
  - Keep a side buffer of suppressed fragment text
  - On the next complete utterance, check if it looks like a continuation
  - If yes: prepend buffered text to the NEW message only (write-once)
  - If no: drop the buffer (user changed topic)
  - Timeout: if buffer is older than ~3s, drop it (user moved on)
  - Pure filler fragments ("um", "uh") are dropped, not merged

Pipeline placement:

    fragment_guard = FragmentGuard(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        fragment_guard,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])
"""

import re
import time
from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


# ── Word categories ──────────────────────────────────────────────────────────

PURE_FILLERS = frozenset({
    "um", "uh", "erm", "hmm", "mm", "mhm", "hm",
    "ah", "eh", "oh",
})

VALID_SHORT_REPLIES = frozenset({
    "yes", "no", "yeah", "yep", "yup", "nah", "nope",
    "okay", "ok", "right", "sure", "fine", "good",
    "sometimes", "never", "always", "rarely", "often",
    "yesterday", "today", "weekly", "daily", "twice",
    "sorry", "thanks", "cheers",
})

DANGLING_CONNECTORS = re.compile(
    r"\b(and|but|so|because|then|if|when|or|that|which|where|"
    r"um|uh|erm|regarding|about|with|for|to|of|in|on|the|your|my|is)\s*[-—…]*\s*$",
    re.IGNORECASE,
)

TRAILING_CUTOFF = re.compile(r"[-—…]+\s*$")

QUESTION_LIKE = re.compile(
    r"\?\s*$"
    r"|^(why|how|what|when|where|who|can|could|would|will|shall|"
    r"is|are|do|did|does|have|has|may|might)\b",
    re.IGNORECASE,
)

# Words that suggest the next utterance is a syntactic continuation
CONTINUATION_STARTERS = re.compile(
    r"^(the|a|an|it|its|they|he|she|this|that|my|your|his|her|"
    r"our|their|some|any|all|each|every|much|more|most|less|"
    r"too|very|really|quite|pretty|just|still|also|"
    r"not|no|yes|"  # continuation after "is" / "are"
    r"\d)",  # numbers continuing a thought
    re.IGNORECASE,
)

# Connectors at the END of a fragment that strongly predict continuation
CONTINUATION_ENDINGS = frozenset({
    "and", "or", "but", "so", "because", "then", "if", "when",
    "about", "with", "for", "to", "of", "in", "on", "at", "by",
    "the", "a", "an", "your", "my", "his", "her", "our", "their",
    "is", "are", "was", "were", "be", "been",
    "not", "very", "really", "quite",
})

# How long to keep buffered fragments before dropping (seconds)
BUFFER_TIMEOUT_SECS = 1.5


# ── Detection logic ──────────────────────────────────────────────────────────

def looks_fragmentary(text: str) -> bool:
    """
    Returns True if `text` looks like an incomplete user turn that should
    NOT trigger an LLM run.
    """
    if not text or not text.strip():
        return True

    t = text.strip()
    t_lower = t.lower()
    t_clean = t_lower.rstrip(".,!?;:—-…'\" ")
    words = re.findall(r"\b\w+\b", t_lower)

    if not words:
        return True

    has_trailing_cutoff = bool(TRAILING_CUTOFF.search(t))

    # Trailing dash = ALWAYS incomplete
    if has_trailing_cutoff:
        return True

    # Escape hatches (only when NOT cut off)
    if QUESTION_LIKE.search(t):
        return False
    if t_clean in VALID_SHORT_REPLIES:
        return False

    # Pure filler
    if t_clean in PURE_FILLERS:
        return True
    if len(words) <= 2 and all(w in PURE_FILLERS for w in words):
        return True

    # Dangling connector
    if len(words) < 8 and DANGLING_CONNECTORS.search(t_lower):
        return True

    # Very short, mostly filler
    real_words = [w for w in words if w not in PURE_FILLERS]
    if len(real_words) <= 1 and len(words) <= 3 and not any(w in VALID_SHORT_REPLIES for w in words):
        return True

    return False


def _clean_fragment_for_merge(text: str) -> str:
    """Strip trailing dashes/ellipsis from a fragment before merging."""
    return re.sub(r"\s*[-—…]+\s*$", "", text).strip()


def _is_pure_filler_text(text: str) -> bool:
    """Check if the entire text is just filler words."""
    words = re.findall(r"\b\w+\b", text.lower())
    return bool(words) and all(w in PURE_FILLERS for w in words)


def _looks_like_continuation(fragment_text: str, next_text: str) -> bool:
    """
    Decide whether `next_text` is likely a continuation of `fragment_text`,
    or a fresh unrelated utterance.

    Returns True  → merge them
    Returns False → drop the fragment, treat next_text as standalone
    """
    frag_clean = _clean_fragment_for_merge(fragment_text).lower()
    next_clean = (next_text or "").strip().lower()

    if not frag_clean or not next_clean:
        return False

    # If the fragment was pure filler, don't merge (nothing useful to prepend)
    if _is_pure_filler_text(fragment_text):
        return False

    # If next utterance is a fresh complete question starting with a question
    # word, it's probably a new attempt, not a continuation
    # UNLESS the fragment ended with a connector that expects an object
    frag_words = re.findall(r"\b\w+\b", frag_clean)
    last_frag_word = frag_words[-1] if frag_words else ""

    if QUESTION_LIKE.search(next_text):
        # A fresh question means the user restarted — don't merge.
        # BUT only count it as a fresh question if it ends with "?" —
        # starting with a question word alone isn't enough.
        # "is inflammation of the tendons." starts with "is" but is
        # clearly a continuation, not a question.
        if re.search(r"\?\s*$", next_text):
            return False
        # Falls through to connector/continuation checks below

    # Strong signal: fragment ends with a connector/preposition
    if last_frag_word in CONTINUATION_ENDINGS:
        return True

    # Moderate signal: next text starts with a continuation word
    if CONTINUATION_STARTERS.match(next_clean):
        return True

    # Moderate signal: next text starts lowercase (not a new sentence)
    if next_clean[0].islower():
        return True

    return False


# ── Pipecat FrameProcessor ───────────────────────────────────────────────────

class FragmentGuard(FrameProcessor):
    """
    Safe fragment accumulator.

    - Suppresses LLMRunFrame for fragments (no LLM call, no TTS, no audio)
    - Stores fragment text in a side buffer with timestamp
    - Does NOT delete any context messages
    - Does NOT mutate older context messages
    - On next complete utterance, checks if it's a plausible continuation
    - If yes: prepends cleaned fragment text into the NEWEST user message
      (the one just created by the aggregator — write-once, minimal mutation)
    - If no: drops the buffer (user changed topic or restarted)
    - Buffer expires after BUFFER_TIMEOUT_SECS (stale fragments are dropped)
    - Pure filler fragments ("um", "uh") are never merged, only dropped
    """

    def __init__(self, context, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        # Buffer: list of (text, timestamp) tuples
        self._fragment_buffer: list[tuple[str, float]] = []
        self._suppressed_count = 0
        self._merged_count = 0
        self._dropped_count = 0
        self._passed_count = 0

    def _get_last_user_message(self) -> dict | None:
        """Find the last user message dict in context."""
        for msg in reversed(self._context.messages):
            if msg.get("role") == "user":
                return msg
        return None

    def _expire_stale_fragments(self):
        """Drop any buffered fragments older than the timeout."""
        if not self._fragment_buffer:
            return
        now = time.time()
        before = len(self._fragment_buffer)
        self._fragment_buffer = [
            (text, ts) for text, ts in self._fragment_buffer
            if (now - ts) < BUFFER_TIMEOUT_SECS
        ]
        expired = before - len(self._fragment_buffer)
        if expired > 0:
            self._dropped_count += expired
            logger.info(
                f"🛡️ FragmentGuard EXPIRED {expired} stale fragment(s) "
                f"(older than {BUFFER_TIMEOUT_SECS}s)"
            )

    def _try_merge_buffer(self, current_msg: dict) -> bool:
        """
        Attempt to merge buffered fragments into the current user message.
        Only merges if the current text looks like a continuation.
        Only modifies the CURRENT (newest) message — never older ones.

        Returns True if merge happened, False if buffer was dropped.
        """
        if not self._fragment_buffer:
            return False

        current_text = (current_msg.get("content") or "").strip()

        # Check continuation plausibility using the LAST buffered fragment
        last_fragment_text = self._fragment_buffer[-1][0]

        if _looks_like_continuation(last_fragment_text, current_text):
            # Merge: clean fragments and prepend to current message
            cleaned = []
            for frag_text, _ in self._fragment_buffer:
                c = _clean_fragment_for_merge(frag_text)
                if c and not _is_pure_filler_text(c):
                    cleaned.append(c)

            if cleaned:
                merged = " ".join(cleaned + [current_text])
                merged = re.sub(r"\s+", " ", merged).strip()

                self._merged_count += 1
                logger.info(
                    f"🛡️ FragmentGuard MERGE #{self._merged_count}: "
                    f"buffer={[t for t, _ in self._fragment_buffer]!r} "
                    f"+ current={current_text!r} → merged={merged!r}"
                )
                current_msg["content"] = merged
            else:
                logger.debug(
                    f"🛡️ FragmentGuard: buffer was all filler, nothing to merge. "
                    f"Keeping: {current_text!r}"
                )

            self._fragment_buffer.clear()
            return True
        else:
            # Not a continuation — drop the buffer
            self._dropped_count += len(self._fragment_buffer)
            logger.info(
                f"🛡️ FragmentGuard DROP buffer (not a continuation): "
                f"buffer={[t for t, _ in self._fragment_buffer]!r} "
                f"next={current_text!r}"
            )
            self._fragment_buffer.clear()
            return False

    async def process_frame(self, frame, direction: FrameDirection):
        if isinstance(frame, LLMRunFrame) and direction == FrameDirection.DOWNSTREAM:
            # Expire old fragments first
            self._expire_stale_fragments()

            last_msg = self._get_last_user_message()

            if last_msg is not None:
                last_user = (last_msg.get("content") or "").strip()
                is_fragment = looks_fragmentary(last_user)

                if is_fragment:
                    self._suppressed_count += 1
                    logger.info(
                        f"🛡️ FragmentGuard SUPPRESS #{self._suppressed_count} "
                        f"text={last_user!r} "
                        f"cutoff={bool(TRAILING_CUTOFF.search(last_user))} "
                        f"connector={bool(DANGLING_CONNECTORS.search(last_user.lower()))}"
                    )

                    # Buffer the fragment — do NOT touch context
                    self._fragment_buffer.append((last_user, time.time()))

                    # Suppress the LLM run
                    return

                else:
                    # Complete utterance — try merging buffered fragments
                    if self._fragment_buffer:
                        self._try_merge_buffer(last_msg)

                    self._passed_count += 1
                    logger.debug(
                        f"🛡️ FragmentGuard PASS #{self._passed_count} "
                        f"text={(last_msg.get('content') or '').strip()!r}"
                    )

        # Everything else passes through
        await self.push_frame(frame, direction)
