from __future__ import annotations

from enum import Enum


class TimeOfDay(str, Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


# Unlike weather, the day always moves forward in the same order -- no rng,
# just a fixed cycle that wraps from night back to morning.
_ORDER = [TimeOfDay.MORNING, TimeOfDay.AFTERNOON, TimeOfDay.EVENING, TimeOfDay.NIGHT]


def next_time_of_day(current: TimeOfDay) -> TimeOfDay:
    return _ORDER[(_ORDER.index(current) + 1) % len(_ORDER)]
