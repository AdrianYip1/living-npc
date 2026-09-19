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


# How far (map units) an NPC can see other people -- what goes in the
# "who's around" block of its prompt (see render_surroundings()). Wider than
# INTERACTION_RANGE, so there's someone to walk over to, not only someone
# already close enough to talk to.
AWARENESS_RANGE = 40

# The two labeled lines render_surroundings() writes. Exposed so anything
# reading the prompt back (the chatty mock backend) parses the same format.
IN_REACH_LABEL = "Close enough to talk to"
IN_VIEW_LABEL = "Further off, but in view"
BUSY_MARK = " (busy talking)"


def render_surroundings(me: tuple[float, float], others: list[tuple[str, tuple[float, float], bool]]) -> str:
    """Who else is within AWARENESS_RANGE of `me`, as a prompt block: one
    line for people within INTERACTION_RANGE (anyone you could start a
    conversation with right now) and one for the rest. `others` is
    (name, position, busy) for everyone else on the map. Empty string when
    no one's around.
    """
    in_reach, in_view = [], []
    for name, position, busy in sorted(others, key=lambda other: distance(me, other[1])):
        gap = distance(me, position)
        if gap > AWARENESS_RANGE:
            continue
        entry = f"{name} at ({position[0]:.0f}, {position[1]:.0f})" + (BUSY_MARK if busy else "")
        (in_reach if gap <= INTERACTION_RANGE else in_view).append(entry)
    lines = []
    if in_reach:
        lines.append(f"{IN_REACH_LABEL}: " + "; ".join(in_reach) + ".")
    if in_view:
        lines.append(f"{IN_VIEW_LABEL}: " + "; ".join(in_view) + ".")
    return "\n".join(lines)


# How close (map units) one NPC stops to another -- well inside
# INTERACTION_RANGE, so walking up to someone still gets you close enough to
# talk, but far enough apart that the two don't draw on top of each other.
PERSONAL_SPACE = 4


def keep_personal_space(
    destination: tuple[int, int], start: tuple[float, float], others: list[tuple[float, float]]
) -> tuple[int, int]:
    """`destination`, moved just far enough (PERSONAL_SPACE) from each point
    in `others` -- where other NPCs stand or are headed -- that the walker
    won't end up on top of anyone. Each push is back toward `start`, so
    walking up to someone stops in front of them, on your side.
    """
    x, y = float(destination[0]), float(destination[1])
    for _ in range(3):  # a push away from one person can land near another
        crowded = False
        for ox, oy in others:
            gap = math.hypot(x - ox, y - oy)
            if gap >= PERSONAL_SPACE - 0.5:
                continue
            crowded = True
            # Straight out from them -- or, when aiming right at their
            # spot, back toward where the walker's coming from (or any
            # fixed direction if the walker is standing there too).
            dx, dy = x - ox, y - oy
            if math.hypot(dx, dy) < 1e-6:
                dx, dy = start[0] - ox, start[1] - oy
            if math.hypot(dx, dy) < 1e-6:
                dx, dy = 1.0, 0.0
            norm = math.hypot(dx, dy)
            x, y = ox + dx / norm * PERSONAL_SPACE, oy + dy / norm * PERSONAL_SPACE
        if not crowded:
            break
    return clamp_coordinate(round(x)), clamp_coordinate(round(y))


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
