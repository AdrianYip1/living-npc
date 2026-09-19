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
INTERACTION_RANGE = 16


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# How far (map units) an NPC can see other people -- what goes in the
# "who's around" block of its prompt (see render_surroundings()). The town is
# small enough to take in at a glance, so: everyone.
AWARENESS_RANGE = math.inf

# The two labeled lines render_surroundings() writes. Exposed so anything
# reading the prompt back (e.g. tests) matches the same format.
IN_REACH_LABEL = "Close enough to talk to"
IN_VIEW_LABEL = "Further off"
BUSY_MARK = " (busy talking)"


def render_surroundings(
    me: tuple[float, float],
    others: list[tuple[str, tuple[float, float], bool]],
    *,
    within: float = AWARENESS_RANGE,
) -> str:
    """Who else is within `within` of `me`, as a prompt block: one
    line for people within INTERACTION_RANGE (anyone you could start a
    conversation with right now) and one for the rest. `others` is
    (name, position, busy) for everyone else on the map. Empty string when
    no one's around.
    """
    in_reach, in_view = [], []
    for name, position, busy in sorted(others, key=lambda other: distance(me, other[1])):
        gap = distance(me, position)
        if gap > within:
            continue
        entry = f"{name} at ({position[0]:.0f}, {position[1]:.0f})" + (BUSY_MARK if busy else "")
        (in_reach if gap <= INTERACTION_RANGE else in_view).append(entry)
    lines = []
    if in_reach:
        lines.append(f"{IN_REACH_LABEL}: " + "; ".join(in_reach) + ".")
    if in_view:
        lines.append(f"{IN_VIEW_LABEL}: " + "; ".join(in_view) + ".")
    return "\n".join(lines)


# How close (map units) one NPC stops to another: a little more than an
# icon's width on the mini-map (NPC_RADIUS 32px / WORLD_SCALE 6, doubled,
# in mini_map/static/app.js), so neighbors never draw on top of each other,
# yet still inside INTERACTION_RANGE -- walking up to someone gets you
# close enough to talk.
PERSONAL_SPACE = 12


def keep_personal_space(
    destination: tuple[int, int], start: tuple[float, float], others: list[tuple[float, float]]
) -> tuple[int, int]:
    """`destination`, unless it's crowded -- someone in `others` (where other
    NPCs stand or are headed) is within PERSONAL_SPACE of it. Then the
    closest free spot on a ring PERSONAL_SPACE out from it, starting on the
    walker's side (coming from `start`) and working around both ways: a
    second visitor to someone stands beside the first, not on them, and
    both stay in talking range of whoever they came to see. If the whole
    ring is taken, the spot on it with the most room.
    """

    def room(point: tuple[float, float]) -> float:
        return min((math.hypot(point[0] - ox, point[1] - oy) for ox, oy in others), default=math.inf)

    if room(destination) >= PERSONAL_SPACE - 0.5:
        return destination
    tx, ty = destination
    base = math.atan2(start[1] - ty, start[0] - tx) if math.hypot(start[0] - tx, start[1] - ty) > 1e-6 else 0.0
    candidates = []
    for step in range(7):  # 30-degree steps, both ways round: the full ring
        for sign in ((1,) if step in (0, 6) else (1, -1)):
            angle = base + sign * step * math.pi / 6
            candidates.append(
                (
                    clamp_coordinate(round(tx + PERSONAL_SPACE * math.cos(angle))),
                    clamp_coordinate(round(ty + PERSONAL_SPACE * math.sin(angle))),
                )
            )
    for candidate in candidates:
        if room(candidate) >= PERSONAL_SPACE - 0.5:
            return candidate
    return max(candidates, key=room)


# NPC walking physics, in map units -- the player's own MAX_SPEED / ACCEL /
# DECEL from mini_map/static/app.js (260 / 900 / 1400, in canvas units),
# divided by its WORLD_SCALE of 6 canvas units per map unit, so NPCs move
# exactly like the player does. Keep the two in sync.
NPC_MAX_SPEED = 260 / 6  # map units/sec
NPC_ACCEL = 900 / 6  # map units/sec^2 while speeding up / turning
NPC_DECEL = 1400 / 6  # map units/sec^2 while braking
# While an NPC is deciding what to do mid-walk (see NPCRegistry.
# step_movement), it slows to this fraction of NPC_MAX_SPEED rather than
# stopping dead -- a glance, not a freeze. Mirrored as HESITATE_SPEED_FACTOR
# in mini_map/static/app.js; keep the two in sync.
HESITATE_SPEED_FACTOR = 0.25


def step_toward(
    position: tuple[float, float],
    velocity: tuple[float, float],
    destination: tuple[float, float] | None,
    dt: float,
    max_speed: float = NPC_MAX_SPEED,
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
    desired_speed = min(max_speed, math.sqrt(2 * NPC_DECEL * dist))
    want_x, want_y = to_x / dist * desired_speed, to_y / dist * desired_speed
    dvx, dvy = want_x - vx, want_y - vy
    dv = math.hypot(dvx, dvy)
    max_dv = (NPC_DECEL if desired_speed < speed else NPC_ACCEL) * dt
    if dv > max_dv:
        dvx, dvy = dvx / dv * max_dv, dvy / dv * max_dv
    vx, vy = vx + dvx, vy + dvy

    return (x + vx * dt, y + vy * dt), (vx, vy), False
