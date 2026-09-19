from __future__ import annotations

from enum import Enum

MINUTES_PER_DAY = 24 * 60


class TimeOfDay(str, Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


# The day divided into four equal 6-hour quarters, each keyed by its start
# minute. Checked in order below, so NIGHT (starting at 0) is always the
# fallback for any minute not covered by a later start.
_PHASE_STARTS: list[tuple[int, TimeOfDay]] = [
    (0, TimeOfDay.NIGHT),
    (6 * 60, TimeOfDay.MORNING),
    (12 * 60, TimeOfDay.AFTERNOON),
    (18 * 60, TimeOfDay.EVENING),
]


def phase_for_minute(minute_of_day: int) -> TimeOfDay:
    minute_of_day %= MINUTES_PER_DAY
    phase = TimeOfDay.NIGHT
    for start, candidate in _PHASE_STARTS:
        if minute_of_day >= start:
            phase = candidate
    return phase


def format_clock(minute_of_day: int) -> str:
    minute_of_day %= MINUTES_PER_DAY
    return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"
