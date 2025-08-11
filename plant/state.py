from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, List, Optional

from loguru import logger

from .metrics_client import PlantMetricsSample


@dataclass(slots=True)
class PlantSummary:
    latest: Optional[PlantMetricsSample]
    seconds_since_update: Optional[float]
    vpd_kpa: Optional[float]
    temperature_status: str
    humidity_status: str
    co2_status: str
    stress_risk: bool
    co2_relative_to_ambient_low: Optional[bool]
    co2_trend_ppm_per_min: Optional[float]
    humidity_trend_pct_per_min: Optional[float]
    temperature_trend_c_per_min: Optional[float]
    # Ready-to-use short phrases for the demo Q&A
    sleep_assessment_text: str
    current_feel_text: str
    productivity_assessment_text: str


class PlantMetricsStore:
    def __init__(self, *, maxlen: int = 720, ambient_co2_baseline_ppm: float = 600.0):
        self._samples: Deque[PlantMetricsSample] = deque(maxlen=maxlen)
        self._ambient_co2_baseline_ppm = ambient_co2_baseline_ppm

    def update(self, sample: PlantMetricsSample) -> None:
        self._samples.append(sample)

    def latest(self) -> Optional[PlantMetricsSample]:
        return self._samples[-1] if self._samples else None

    def window(self, since: timedelta) -> List[PlantMetricsSample]:
        if not self._samples:
            return []
        cutoff = datetime.now(timezone.utc) - since
        return [s for s in list(self._samples) if s.timestamp >= cutoff]

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    def _trend_per_min(self, series: List[float], seconds: List[float]) -> Optional[float]:
        # Simple slope via first-last over minutes to keep robust and cheap
        if len(series) < 2 or len(series) != len(seconds):
            return None
        dy = series[-1] - series[0]
        dt_min = (seconds[-1] - seconds[0]) / 60.0
        if dt_min <= 0:
            return None
        return dy / dt_min

    @staticmethod
    def _calc_vpd_kpa(temp_c: float, rh_pct: float) -> float:
        # Tetens formula for saturation vapor pressure (kPa)
        es = 0.6108 * math.exp(17.27 * temp_c / (temp_c + 237.3))
        ea = es * (rh_pct / 100.0)
        return max(0.0, es - ea)

    def summarize(self) -> PlantSummary:
        latest = self.latest()
        now = datetime.now(timezone.utc)
        seconds_since_update = (now - latest.timestamp).total_seconds() if latest else None

        # 10-min window for trends
        win = self.window(timedelta(minutes=10))
        times = [s.timestamp.timestamp() for s in win]
        temps = [s.temperature_c for s in win]
        hums = [s.humidity_pct for s in win]
        co2s = [s.co2_ppm for s in win]

        temp_trend = self._trend_per_min(temps, times) if temps else None
        hum_trend = self._trend_per_min(hums, times) if hums else None
        co2_trend = self._trend_per_min(co2s, times) if co2s else None

        vpd = self._calc_vpd_kpa(latest.temperature_c, latest.humidity_pct) if latest else None

        # Status heuristics
        temperature_status = "unknown"
        humidity_status = "unknown"
        co2_status = "unknown"
        stress_risk = False
        co2_rel_low: Optional[bool] = None

        if latest:
            t = latest.temperature_c
            h = latest.humidity_pct
            c = latest.co2_ppm

            temperature_status = "hot" if t > 30 else "comfy" if 18 <= t <= 28 else "cool"
            if h < 25:
                humidity_status = "very_dry"
            elif h < 35:
                humidity_status = "dry"
            elif h > 80:
                humidity_status = "humid"
            else:
                humidity_status = "ideal"

            if c > 1500:
                co2_status = "very_stale"
            elif c > 1200:
                co2_status = "stale"
            elif c < 450:
                co2_status = "fresh"
            else:
                co2_status = "normal"

            stress_risk = (h < 35) and (t >= 28)
            try:
                co2_rel_low = c < (self._ambient_co2_baseline_ppm - 50)
            except Exception:
                co2_rel_low = None

        # Night window (22:00-06:00 local) in last 8 hours for rough sleep assessment
        last8 = self.window(timedelta(hours=8))
        def _is_night(ts: datetime) -> bool:
            hour = ts.astimezone().hour
            return hour >= 22 or hour < 6
        night_co2 = [s.co2_ppm for s in last8 if _is_night(s.timestamp)]
        night_h = [s.humidity_pct for s in last8 if _is_night(s.timestamp)]
        night_t = [s.temperature_c for s in last8 if _is_night(s.timestamp)]

        # Sleep assessment
        if night_co2:
            avg_night_co2 = self._mean(night_co2) or 0.0
            sleep_assessment = (
                "I slept great. My indicators show healthy elevated CO2 levels and I slept like a baby."
                if avg_night_co2 >= (self._ambient_co2_baseline_ppm - 25)
                else "I rested fine. CO2 stayed moderate overnight, and I felt calm."
            )
        else:
            sleep_assessment = (
                "I slept well. Things felt calm and steady through the night."
            )

        # Current feel assessment
        if stress_risk:
            current_feel = (
                "Honestly, temperatures are increasing and I feel like I need water. So, I'm getting a bit stressed."
            )
        elif temperature_status == "hot":
            current_feel = "It's quite warm—I could use a cooler breeze or some shade."
        elif humidity_status in ("dry", "very_dry"):
            current_feel = "A bit dry—I could use some water to stay comfy."
        else:
            current_feel = "I feel comfortable right now—thank you for checking on me!"

        # Productivity assessment (photosynthesis proxy via CO2 drawdown vs ambient)
        if co2_rel_low is True:
            productivity = (
                "My sensors indicate low CO2 concentrations around my leaves. Makes sense because I'm actively photosynthesizing!"
            )
        elif co2_trend is not None and co2_trend < -2.0:
            productivity = (
                "CO2 is dropping around me, which matches active photosynthesis—I'm doing my leafy best!"
            )
        else:
            productivity = (
                "I'm doing my best. If you open a window or give me more light, I can work even harder."
            )

        return PlantSummary(
            latest=latest,
            seconds_since_update=seconds_since_update,
            vpd_kpa=vpd,
            temperature_status=temperature_status,
            humidity_status=humidity_status,
            co2_status=co2_status,
            stress_risk=stress_risk,
            co2_relative_to_ambient_low=co2_rel_low,
            co2_trend_ppm_per_min=co2_trend,
            humidity_trend_pct_per_min=hum_trend,
            temperature_trend_c_per_min=temp_trend,
            sleep_assessment_text=sleep_assessment,
            current_feel_text=current_feel,
            productivity_assessment_text=productivity,
        )

    def to_result_dict(self, *, units: str = "metric") -> Dict:
        summary = self.summarize()
        latest = summary.latest
        if latest is None:
            return {"available": False, "message": "No sensor data available yet."}

        temp = latest.temperature_c
        hum = latest.humidity_pct
        co2 = latest.co2_ppm

        if units == "imperial":
            temp_out = temp * 9.0 / 5.0 + 32.0
            temp_key = "temperature_f"
        else:
            temp_out = temp
            temp_key = "temperature_c"

        result = {
            "available": True,
            "latest": {
                temp_key: round(temp_out, 2),
                "humidity_pct": round(hum, 2),
                "co2_ppm": round(co2, 1),
                "timestamp": latest.timestamp.isoformat(),
                "seconds_since_update": summary.seconds_since_update,
            },
            "derived": {
                "vpd_kpa": round(summary.vpd_kpa, 3) if summary.vpd_kpa is not None else None,
                "temperature_status": summary.temperature_status,
                "humidity_status": summary.humidity_status,
                "co2_status": summary.co2_status,
                "stress_risk": summary.stress_risk,
                "co2_relative_to_ambient_low": summary.co2_relative_to_ambient_low,
                "trends": {
                    "co2_trend_ppm_per_min": summary.co2_trend_ppm_per_min,
                    "humidity_trend_pct_per_min": summary.humidity_trend_pct_per_min,
                    "temperature_trend_c_per_min": summary.temperature_trend_c_per_min,
                },
            },
            "phrases": {
                "sleep": summary.sleep_assessment_text,
                "current_feel": summary.current_feel_text,
                "productivity": summary.productivity_assessment_text,
            },
        }
        return result 