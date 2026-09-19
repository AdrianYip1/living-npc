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
    weather roll, a flat spawn-chance coin flip, name-pool sampling for
    travelers) -- NOT an LLM call. This is a placeholder for a real build:
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
    """

    def __init__(
        self,
        weather: Weather = Weather.CLEAR,
        traveler_chance: float = 0.15,
        start_minute: int = 8 * 60,  # 08:00 -- lands in morning by default
        # 1 minute -- the finest step format_clock() can even show, so the
        # clock always counts up smoothly. How fast that happens in real
        # time is tick_interval_s's job (see mini_map.Simulation), not
        # this: bumping this up trades smoothness for bigger jumps instead.
        minutes_per_tick: int = 1,
        seed: int | None = None,
    ) -> None:
        self.weather = weather
        self.traveler_chance = traveler_chance
        self.minute_of_day = start_minute % MINUTES_PER_DAY
        self.minutes_per_tick = minutes_per_tick
        self._rng = random.Random(seed)
        self.temperature = next_temperature(self.time_of_day, self.weather, self._rng)
        self._last_weather_hour = self.minute_of_day // 60

    @property
    def time_of_day(self) -> TimeOfDay:
        return phase_for_minute(self.minute_of_day)

    @property
    def clock(self) -> str:
        return format_clock(self.minute_of_day)

    def tick(self) -> EnvironmentEvent:
        self.minute_of_day = (self.minute_of_day + self.minutes_per_tick) % MINUTES_PER_DAY
        current_hour = self.minute_of_day // 60

        changed = False
        if current_hour != self._last_weather_hour:
            self._last_weather_hour = current_hour
            new_weather = next_weather(self.weather, self._rng)
            changed = new_weather != self.weather
            self.weather = new_weather
            self.temperature = next_temperature(self.time_of_day, self.weather, self._rng)

        travelers_arrived = []
        if self._rng.random() < self.traveler_chance:
            travelers_arrived.append(invent_traveler(self._rng))

        return EnvironmentEvent(
            weather=self.weather,
            weather_changed=changed,
            time_of_day=self.time_of_day,
            clock=self.clock,
            temperature=self.temperature,
            travelers_arrived=travelers_arrived,
        )
