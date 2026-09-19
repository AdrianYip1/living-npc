from __future__ import annotations

# The minimap is a 200x200 square centered on the origin -- every NPC's
# home, workplace, and current position live inside this range.
MAP_MIN = -100
MAP_MAX = 100


def clamp_coordinate(value: int) -> int:
    return max(MAP_MIN, min(MAP_MAX, value))
