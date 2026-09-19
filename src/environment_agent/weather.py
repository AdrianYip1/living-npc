from __future__ import annotations

import random
from enum import Enum


class Weather(str, Enum):
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"


# Deterministic stand-in for what should eventually be an LLM call ("given
# yesterday's weather, season, recent history, what's today's weather?").
# For now: a fixed weighted-transition table per current state, so weather
# still drifts and holds together (clear rarely jumps straight to a storm)
# without any real judgment behind it.
_TRANSITIONS: dict[Weather, dict[Weather, float]] = {
    Weather.CLEAR: {Weather.CLEAR: 0.6, Weather.CLOUDY: 0.3, Weather.RAINY: 0.1, Weather.STORMY: 0.0},
    Weather.CLOUDY: {Weather.CLEAR: 0.3, Weather.CLOUDY: 0.4, Weather.RAINY: 0.25, Weather.STORMY: 0.05},
    Weather.RAINY: {Weather.CLEAR: 0.1, Weather.CLOUDY: 0.35, Weather.RAINY: 0.45, Weather.STORMY: 0.1},
    Weather.STORMY: {Weather.CLEAR: 0.05, Weather.CLOUDY: 0.25, Weather.RAINY: 0.4, Weather.STORMY: 0.3},
}


def next_weather(current: Weather, rng: random.Random) -> Weather:
    """One weighted roll off `current`'s row. Swap this body for an LLM
    call later; callers only ever see "given current weather, what's next".
    """
    options = _TRANSITIONS[current]
    return rng.choices(list(options.keys()), weights=list(options.values()), k=1)[0]
