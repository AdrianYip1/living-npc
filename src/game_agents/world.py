from __future__ import annotations

import math

# The minimap is a 200x200 square centered on the origin -- every NPC's
# home, workplace, and current position live inside this range.
MAP_MIN = -100
MAP_MAX = 100


def clamp_coordinate(value: int) -> int:
    return max(MAP_MIN, min(MAP_MAX, value))

# How close (in map units, straight-line) two NPCs must be to talk to or
# trade with each other.
INTERACTION_RANGE = 10


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# NPC walking physics, in map units -- the player's own MAX_SPEED / ACCEL /
# DECEL from mini_map/static/app.js (260 / 900 / 1400, in canvas units),
# divided by its WORLD_SCALE of 6 canvas units per map unit, so NPCs move
# exactly like the player does. Keep the two in sync.
NPC_MAX_SPEED = 260 / 6  # map units/sec
NPC_ACCEL = 900 / 6  # map units/sec^2 while speeding up / turning
NPC_DECEL = 1400 / 6  # map units/sec^2 while braking


def step_toward(
    position: tuple[float, float],
    velocity: tuple[float, float],
    destination: tuple[float, float] | None,
    dt: float,
) -> tuple[tuple[float, float], tuple[float, float], bool]:
    """One physics step of walking toward `destination` (or coasting to a
    stop if it's None), with the same momentum the player has: velocity
    changes by at most NPC_ACCEL (or NPC_DECEL when braking) per second,
    capped at NPC_MAX_SPEED, and the walker starts braking just early
    enough to stop on the spot instead of overshooting it.

    Returns (position, velocity, arrived).
    """
    x, y = position
    vx, vy = velocity
    speed = math.hypot(vx, vy)

    if destination is None:
        if speed > 0:
            scale = max(0.0, speed - NPC_DECEL * dt) / speed
            vx, vy = vx * scale, vy * scale
        return (x + vx * dt, y + vy * dt), (vx, vy), False

    to_x, to_y = destination[0] - x, destination[1] - y
    dist = math.hypot(to_x, to_y)
    # Close enough that this step would reach (or pass) it: land on it.
    if dist <= max(speed * dt, 1e-6):
        return (float(destination[0]), float(destination[1])), (0.0, 0.0), True

    # Fastest speed that can still brake to zero within `dist`.
    desired_speed = min(NPC_MAX_SPEED, math.sqrt(2 * NPC_DECEL * dist))
    want_x, want_y = to_x / dist * desired_speed, to_y / dist * desired_speed
    dvx, dvy = want_x - vx, want_y - vy
    dv = math.hypot(dvx, dvy)
    max_dv = (NPC_DECEL if desired_speed < speed else NPC_ACCEL) * dt
    if dv > max_dv:
        dvx, dvy = dvx / dv * max_dv, dvy / dv * max_dv
    vx, vy = vx + dvx, vy + dvy

    return (x + vx * dt, y + vy * dt), (vx, vy), False
