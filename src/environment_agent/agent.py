from __future__ import annotations

import random
from dataclasses import dataclass, field

from .temperature import next_temperature
from .time_of_day import MINUTES_PER_DAY, TimeOfDay, format_clock, phase_for_minute
from .travelers import Traveler, invent_traveler
from .weather import Weather, next_weather


@dataclass
class EnvironmentEvent:
    """One tick's worth of environmental decisions -- the environment
    agent's equivalent of the NPC Agent's TurnResult: a single, one-shot
    output, no lookahead or schedule (see the project's core invariants).
    """

    weather: Weather
    weather_changed: bool
    time_of_day: TimeOfDay
    clock: str  # "HH:MM" in-game time, e.g. "08:15" -- see time_of_day.format_clock
    temperature: int  # degrees Fahrenheit -- see temperature.next_temperature
    travelers_arrived: list[Traveler] = field(default_factory=list)


class EnvironmentAgent:
    """Decides weather, how often travelers pass through, and invents
    traveler identities at spawn time. Kept separate from NPC agents, and
    from anything an NPC's own Agent does (see the project's core
    invariants) -- this is the thing that should eventually re-trigger NPC
    idle behavior on a tick, not the other way around.

    Everything in this class is currently deterministic (a weighted random
    weather roll, a per-day traveler count and arrival times, name-pool
    sampling for travelers) -- NOT an LLM call. This is a placeholder for a real build:
    tick() already has the shape an LLM turn would have (current state in,
    one decision out, no memory of "what's next"), so swapping the
    deterministic internals for an LLM call later shouldn't change how
    callers use it.

    The in-game clock (minute_of_day) is the one exception to "no memory of
    what's next": time always moves forward by a fixed amount each tick, so
    it's tracked directly rather than rolled -- time_of_day is a derived
    view of it (see time_of_day.phase_for_minute), not a separate value
    advanced on its own.

    Temperature is derived from that beat's time_of_day and weather (see
    temperature.next_temperature), not rolled independently -- there's no
    such thing as a 40-degree afternoon right after a 90-degree one.

    Weather (and, since it depends on weather, temperature) only actually
    rolls once per in-game hour, tracked by _last_weather_hour -- not once
    per tick. minutes_per_tick can be as fine as 1 game-minute for a smooth
    clock, and ticks can fire as fast as tick_interval_s allows; weather
    changing at that same pace would make it flicker unrealistically fast
    regardless of how those two are tuned.

    Travelers are the other exception: at the start of each in-game day
    (and on construction, for whatever's left of the starting day) the
    agent rolls how many travelers will come that day, somewhere in
    travelers_per_day, and a random minute for each -- see _plan_day().
    tick() then just releases whichever planned arrivals its time step
    passed. Who each traveler is still gets invented at arrival, not at
    planning time.
    """

    def __init__(
        self,
        weather: Weather = Weather.CLEAR,
        # Inclusive (min, max) arrivals per in-game day.
        travelers_per_day: tuple[int, int] = (2, 6),
        start_minute: int = 8 * 60,  # 08:00 -- lands in morning by default
        # 1 minute -- the finest step format_clock() can even show, so the
        # clock always counts up smoothly. How fast that happens in real
        # time is tick_interval_s's job (see mini_map.Simulation), not
        # this: bumping this up trades smoothness for bigger jumps instead.
        minutes_per_tick: int = 1,
        seed: int | None = None,
    ) -> None:
        low, high = travelers_per_day
        if not 0 <= low <= high:
            raise ValueError(f"travelers_per_day must satisfy 0 <= min <= max, got {travelers_per_day}")
        self.weather = weather
        self.travelers_per_day = travelers_per_day
        self.minute_of_day = start_minute % MINUTES_PER_DAY
        self.minutes_per_tick = minutes_per_tick
        self._rng = random.Random(seed)
        self.temperature = next_temperature(self.time_of_day, self.weather, self._rng)
        self._last_weather_hour = self.minute_of_day // 60

        # Game-minutes elapsed since midnight of day 0 -- unlike
        # minute_of_day it never wraps, so planned arrivals from different
        # days can share one sorted list.
        self._elapsed_minutes = self.minute_of_day
        self._planned_through_day = 0
        self._planned_arrivals: list[int] = self._plan_day(0, after=self._elapsed_minutes)

    def _plan_day(self, day: int, after: int = -1) -> list[int]:
        """Rolls day `day`'s traveler count and arrival minutes, as absolute
        elapsed-minute values. Arrivals at or before `after` are dropped --
        used for the starting day, whose earlier hours already "happened".
        """
        count = self._rng.randint(*self.travelers_per_day)
        day_start = day * MINUTES_PER_DAY
        times = sorted(day_start + self._rng.randrange(MINUTES_PER_DAY) for _ in range(count))
        return [t for t in times if t > after]

    @property
    def planned_arrivals_today(self) -> list[str]:
        """Clock times ("HH:MM") of arrivals still to come today -- for
        debugging/inspection only; callers shouldn't plan around it.
        """
        day_end = (self._elapsed_minutes // MINUTES_PER_DAY + 1) * MINUTES_PER_DAY
        return [format_clock(t) for t in self._planned_arrivals if t < day_end]

    @property
    def elapsed_minutes(self) -> int:
        """Game-minutes since midnight of day 0 -- never wraps, unlike
        minute_of_day, so it can measure how long something has lasted.
        """
        return self._elapsed_minutes

    @property
    def time_of_day(self) -> TimeOfDay:
        return phase_for_minute(self.minute_of_day)

    @property
    def clock(self) -> str:
        return format_clock(self.minute_of_day)

    def tick(self) -> EnvironmentEvent:
        self._elapsed_minutes += self.minutes_per_tick
        self.minute_of_day = self._elapsed_minutes % MINUTES_PER_DAY
        current_hour = self.minute_of_day // 60

        changed = False
        if current_hour != self._last_weather_hour:
            self._last_weather_hour = current_hour
            new_weather = next_weather(self.weather, self._rng)
            changed = new_weather != self.weather
            self.weather = new_weather
            self.temperature = next_temperature(self.time_of_day, self.weather, self._rng)

        # A loop, not an if: one big tick can cross several midnights.
        current_day = self._elapsed_minutes // MINUTES_PER_DAY
        while self._planned_through_day < current_day:
            self._planned_through_day += 1
            self._planned_arrivals.extend(self._plan_day(self._planned_through_day))

        due = 0
        while due < len(self._planned_arrivals) and self._planned_arrivals[due] <= self._elapsed_minutes:
            due += 1
        del self._planned_arrivals[:due]
        travelers_arrived = [invent_traveler(self._rng) for _ in range(due)]

        return EnvironmentEvent(
            weather=self.weather,
            weather_changed=changed,
            time_of_day=self.time_of_day,
            clock=self.clock,
            temperature=self.temperature,
            travelers_arrived=travelers_arrived,
        )
