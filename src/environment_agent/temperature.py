from __future__ import annotations

import random

from .time_of_day import TimeOfDay
from .weather import Weather

# Deterministic stand-in for what should eventually be an LLM call ("given
# time of day, weather, and season, what's the temperature?"). For now: a
# fixed baseline per time of day, nudged by the current weather, plus a
# little random jitter so back-to-back ticks aren't identical. Degrees
# Fahrenheit, no seasons yet.
_BASE_BY_TIME_OF_DAY: dict[TimeOfDay, int] = {
    TimeOfDay.NIGHT: 45,
    TimeOfDay.MORNING: 58,
    TimeOfDay.AFTERNOON: 75,
    TimeOfDay.EVENING: 62,
}

_OFFSET_BY_WEATHER: dict[Weather, int] = {
    Weather.CLEAR: 3,
    Weather.CLOUDY: 0,
    Weather.RAINY: -5,
    Weather.STORMY: -9,
}

_JITTER = 2  # +/- degrees of random noise per tick

MIN_TEMPERATURE = 20
MAX_TEMPERATURE = 100


def next_temperature(time_of_day: TimeOfDay, weather: Weather, rng: random.Random) -> int:
    """Degrees Fahrenheit, given the tick's time of day and weather. Swap
    this body for an LLM call later; callers only ever see "given time of
    day and weather, what's the temperature" in, a single value out.
    """
    base = _BASE_BY_TIME_OF_DAY[time_of_day] + _OFFSET_BY_WEATHER[weather]
    jittered = base + rng.randint(-_JITTER, _JITTER)
    return max(MIN_TEMPERATURE, min(MAX_TEMPERATURE, jittered))
