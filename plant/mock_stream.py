from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
from datetime import datetime, timezone
from typing import AsyncIterator

from aiohttp import web, ClientConnectionError


def _iso_now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


async def generate_samples(
    interval_sec: float,
    start_temp_c: float,
    start_humidity_pct: float,
    start_co2_ppm: float,
) -> AsyncIterator[bytes]:
    t = 0.0
    temp = start_temp_c
    hum = start_humidity_pct
    co2 = start_co2_ppm
    while True:
        # Mild random walk with small sinusoid to look alive
        t += interval_sec
        temp += random.uniform(-0.05, 0.05) + 0.02 * math.sin(t / 15.0)
        hum += random.uniform(-0.2, 0.2) + 0.3 * math.sin(t / 10.0)
        co2 += random.uniform(-5, 5) + 2.0 * math.sin(t / 20.0)

        sample = {
            "timestamp": _iso_now_z(),
            "co2_ppm": max(350.0, round(co2, 1)),
            "temperature_c": round(temp, 5),
            "humidity_pct": max(0.0, round(hum, 5)),
        }
        line = json.dumps(sample) + "\n"
        yield line.encode("utf-8")
        await asyncio.sleep(interval_sec)


async def stream_handler(request: web.Request) -> web.StreamResponse:
    interval_sec = float(request.app["interval_sec"])  # type: ignore[index]
    start_temp_c = float(request.app["start_temp_c"])  # type: ignore[index]
    start_humidity_pct = float(request.app["start_humidity_pct"])  # type: ignore[index]
    start_co2_ppm = float(request.app["start_co2_ppm"])  # type: ignore[index]

    resp = web.StreamResponse(status=200, reason="OK")
    resp.content_type = "application/x-ndjson"
    await resp.prepare(request)

    try:
        async for chunk in generate_samples(
            interval_sec=interval_sec,
            start_temp_c=start_temp_c,
            start_humidity_pct=start_humidity_pct,
            start_co2_ppm=start_co2_ppm,
        ):
            await resp.write(chunk)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError, ClientConnectionError):
        # Client disconnected; end streaming silently
        pass
    except Exception:
        # Swallow other transient errors to keep server robust in dev
        pass
    finally:
        transport = request.transport
        if transport is not None and not transport.is_closing():
            try:
                await resp.write_eof()
            except Exception:
                pass
    return resp


async def health_handler(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


def build_app(args: argparse.Namespace) -> web.Application:
    app = web.Application()
    app["interval_sec"] = args.interval
    app["start_temp_c"] = args.start_temp_c
    app["start_humidity_pct"] = args.start_humidity_pct
    app["start_co2_ppm"] = args.start_co2_ppm
    app.add_routes([
        web.get("/metrics/plant_stream", stream_handler),
        web.get("/healthz", health_handler),
    ])
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock NDJSON plant metrics streamer")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=9099, help="Bind port")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between samples")
    parser.add_argument("--start-temp-c", type=float, default=28.14, help="Start temperature (°C)")
    parser.add_argument("--start-humidity-pct", type=float, default=36.95, help="Start humidity (%)")
    parser.add_argument("--start-co2-ppm", type=float, default=590.0, help="Start eCO2 (ppm)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main() 