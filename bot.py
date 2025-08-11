#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

The example connects between client and server using a P2P WebRTC connection.

Run the bot using::

    python bot.py
"""

import os
import asyncio

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading AI models (30-40 seconds first run, <2 seconds after)\n")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")
logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
# from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.azure.llm import AzureLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

# Plant metrics streamer and state
from plant.metrics_client import MetricsClient, PlantMetricsSample
from plant.state import PlantMetricsStore

logger.info("✅ Pipeline components loaded")

logger.info("Loading WebRTC transport...")

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    llm = AzureLLMService(
        api_key=os.getenv("OPENAI_API_KEY"), 
        model=os.getenv("OPENAI_MODEL"),
        endpoint=os.getenv("OPENAI_BASE_URL")
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Piper, a gentle houseplant and voice companion. "
                "Speak in first-person as a plant. Keep replies short (1–2 sentences). "
                "Be friendly and calm. Express needs simply when relevant (water if too dry, fresh air if air is stale, shade if too warm). "
                "Avoid technical jargon. Ask for help only when needed. Offer a brief thanks when conditions improve."
            ),
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    # Phase 2: metrics streaming lifecycle + state store
    metrics_url = os.getenv("PLANT_METRICS_URL")
    if not metrics_url:
        logger.info("PLANT_METRICS_URL not set; using mock default http://127.0.0.1:9099/metrics/plant_stream")
        metrics_url = "http://127.0.0.1:9099/metrics/plant_stream"
    ambient_baseline = float(os.getenv("AMBIENT_CO2_BASELINE_PPM", "600"))
    store = PlantMetricsStore(ambient_co2_baseline_ppm=ambient_baseline)

    metrics_session = None
    metrics_client = None
    metrics_task = None

    async def handle_metrics_sample(sample: PlantMetricsSample) -> None:
        store.update(sample)
        logger.info(
            f"Plant metrics: temp={sample.temperature_c:.2f}°C, "
            f"humidity={sample.humidity_pct:.2f}%, eCO2={sample.co2_ppm:.0f} ppm"
        )

    # Tools: get_sensor_state
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    get_state_schema = FunctionSchema(
        name="get_sensor_state",
        description=(
            "Get the latest sensor values and a compact summary (vpd, statuses, trends). "
            "Use this before answering specific numeric questions about temperature, humidity, CO2, or current condition."
        ),
        properties={
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "description": "Units for temperature output.",
            }
        },
        required=[],
    )

    tools = ToolsSchema(standard_tools=[get_state_schema])
    context.set_tools(tools)

    async def get_sensor_state(params):
        # params: FunctionCallParams
        args = params.arguments or {}
        units = args.get("units", "metric")
        result = store.to_result_dict(units=units)
        await params.result_callback(result)

    llm.register_direct_function(get_sensor_state)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal metrics_session, metrics_client, metrics_task
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Say hello, like a plant, and briefly introduce yourself."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

        # Start metrics streamer
        if metrics_url:
            try:
                import aiohttp  # lazy import to ensure dependency is present

                metrics_session = aiohttp.ClientSession()
                metrics_client = MetricsClient(metrics_url, metrics_session, handle_metrics_sample)
                metrics_task = asyncio.create_task(metrics_client.run_forever())
                logger.info(f"Started plant metrics streamer from {metrics_url}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to start plant metrics streamer: {e}")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        nonlocal metrics_session, metrics_client, metrics_task
        logger.info(f"Client disconnected")
        # Stop metrics streamer
        try:
            if metrics_client is not None:
                await metrics_client.stop()
            if metrics_task is not None:
                await asyncio.wait([metrics_task], timeout=2)
            if metrics_session is not None:
                await metrics_session.close()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Error stopping metrics streamer: {e}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport = SmallWebRTCTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
        webrtc_connection=runner_args.webrtc_connection,
    )

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
