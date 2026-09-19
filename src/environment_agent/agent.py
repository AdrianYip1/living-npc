from __future__ import annotations

import random
from dataclasses import dataclass, field

from .time_of_day import TimeOfDay, next_time_of_day
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
    """

    def __init__(
        self,
        weather: Weather = Weather.CLEAR,
        traveler_chance: float = 0.15,
        time_of_day: TimeOfDay = TimeOfDay.MORNING,
        seed: int | None = None,
    ) -> None:
        self.weather = weather
        self.traveler_chance = traveler_chance
        self.time_of_day = time_of_day
        self._rng = random.Random(seed)

    def tick(self) -> EnvironmentEvent:
        new_weather = next_weather(self.weather, self._rng)
        changed = new_weather != self.weather
        self.weather = new_weather
        self.time_of_day = next_time_of_day(self.time_of_day)

        travelers_arrived = []
        if self._rng.random() < self.traveler_chance:
            travelers_arrived.append(invent_traveler(self._rng))

        return EnvironmentEvent(
            weather=self.weather,
            weather_changed=changed,
            time_of_day=self.time_of_day,
            travelers_arrived=travelers_arrived,
        )
