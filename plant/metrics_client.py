from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

import aiohttp
from loguru import logger


@dataclass(slots=True)
class PlantMetricsSample:
    timestamp: datetime
    co2_ppm: float
    temperature_c: float
    humidity_pct: float


class MetricsClient:
    """NDJSON metrics streamer for plant readings.

    Expects each line to be a JSON object containing keys:
      - "timestamp" (ISO 8601; Z supported) or omitted
      - "co2_ppm" (float)
      - "temperature_c" (float)
      - "humidity_pct" (float)
    """

    def __init__(
        self,
        url: str,
        session: aiohttp.ClientSession,
        on_sample: Callable[[PlantMetricsSample], Awaitable[None]],
        reconnect_delay_sec: float = 2.0,
    ) -> None:
        self._url = url
        self._session = session
        self._on_sample = on_sample
        self._reconnect_delay_sec = reconnect_delay_sec
        self._stop_event = asyncio.Event()

    async def stop(self) -> None:
        self._stop_event.set()

    async def run_forever(self) -> None:
        while not self._stop_event.is_set():
            try:
                async with self._session.get(self._url, timeout=None) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"MetricsClient: non-200 status {resp.status} from {self._url}"
                        )
                        await self._wait_with_cancel(self._reconnect_delay_sec)
                        continue

                    while not self._stop_event.is_set():
                        raw_line = await resp.content.readline()
                        if not raw_line:
                            break
                        line = raw_line.decode("utf-8").strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            logger.debug(f"MetricsClient: skipping invalid JSON line: {line!r}")
                            continue
                        sample = self._parse_sample(data)
                        if sample is None:
                            continue
                        try:
                            await self._on_sample(sample)
                        except Exception as e:  # noqa: BLE001
                            logger.exception(f"MetricsClient: on_sample callback failed: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.warning(f"MetricsClient: connection error: {e}")
                await self._wait_with_cancel(self._reconnect_delay_sec)

    async def _wait_with_cancel(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    def _parse_sample(self, data: dict) -> Optional[PlantMetricsSample]:
        try:
            ts_raw = data.get("timestamp")
            if isinstance(ts_raw, str):
                ts = self._parse_iso8601(ts_raw)
            else:
                ts = datetime.now(timezone.utc)

            co2 = float(data["co2_ppm"]) if "co2_ppm" in data else None
            temp = float(data["temperature_c"]) if "temperature_c" in data else None
            hum = float(data["humidity_pct"]) if "humidity_pct" in data else None
            if None in (co2, temp, hum):
                logger.debug(
                    f"MetricsClient: missing fields in line; have keys {list(data.keys())}"
                )
                return None
            return PlantMetricsSample(timestamp=ts, co2_ppm=co2, temperature_c=temp, humidity_pct=hum)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"MetricsClient: failed to parse sample: {e}; data={data!r}")
            return None

    @staticmethod
    def _parse_iso8601(value: str) -> datetime:
        # Accept trailing 'Z' as UTC
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value) 