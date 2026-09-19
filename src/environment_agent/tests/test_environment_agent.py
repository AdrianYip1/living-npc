from __future__ import annotations

import random
import unittest

from environment_agent.agent import EnvironmentAgent
from environment_agent.time_of_day import TimeOfDay, next_time_of_day
from environment_agent.travelers import _FIRST_NAMES, _ORIGINS, _REASONS, _TRAITS, invent_traveler
from environment_agent.weather import Weather, next_weather


class NextWeatherTests(unittest.TestCase):
    def test_always_returns_a_known_weather(self):
        rng = random.Random(1)
        for _ in range(200):
            for current in Weather:
                self.assertIn(next_weather(current, rng), set(Weather))

    def test_same_seed_is_reproducible(self):
        a = [next_weather(Weather.CLEAR, random.Random(42)) for _ in range(20)]
        b = [next_weather(Weather.CLEAR, random.Random(42)) for _ in range(20)]
        self.assertEqual(a, b)


class InventTravelerTests(unittest.TestCase):
    def test_fields_come_from_the_word_pools(self):
        traveler = invent_traveler(random.Random(7))

        self.assertIn(traveler.name, _FIRST_NAMES)
        self.assertIn(traveler.origin, _ORIGINS)
        self.assertIn(traveler.reason, _REASONS)
        self.assertEqual(len(traveler.traits), 2)
        self.assertTrue(set(traveler.traits).issubset(set(_TRAITS)))

    def test_traits_are_not_duplicated(self):
        traveler = invent_traveler(random.Random(7))
        self.assertEqual(len(set(traveler.traits)), len(traveler.traits))


class NextTimeOfDayTests(unittest.TestCase):
    def test_cycles_in_a_fixed_order_regardless_of_rng(self):
        current = TimeOfDay.MORNING
        seen = [current]
        for _ in range(4):
            current = next_time_of_day(current)
            seen.append(current)

        self.assertEqual(
            seen,
            [TimeOfDay.MORNING, TimeOfDay.AFTERNOON, TimeOfDay.EVENING, TimeOfDay.NIGHT, TimeOfDay.MORNING],
        )


class EnvironmentAgentTickTests(unittest.TestCase):
    """tick() is the one-shot, no-lookahead entry point standing in for a
    future LLM turn -- these check the deterministic internals behave, not
    that any particular weather or traveler was chosen.
    """

    def test_zero_traveler_chance_never_spawns_anyone(self):
        env = EnvironmentAgent(traveler_chance=0.0, seed=1)
        for _ in range(50):
            self.assertEqual(env.tick().travelers_arrived, [])

    def test_certain_traveler_chance_always_spawns_exactly_one(self):
        env = EnvironmentAgent(traveler_chance=1.0, seed=1)
        for _ in range(20):
            self.assertEqual(len(env.tick().travelers_arrived), 1)

    def test_weather_changed_flag_matches_actual_transitions(self):
        env = EnvironmentAgent(seed=3)
        previous = env.weather
        for _ in range(200):
            event = env.tick()
            self.assertEqual(event.weather_changed, event.weather != previous)
            previous = event.weather

    def test_same_seed_produces_the_same_tick_sequence(self):
        a = EnvironmentAgent(seed=99)
        b = EnvironmentAgent(seed=99)

        a_weathers = [a.tick().weather for _ in range(30)]
        b_weathers = [b.tick().weather for _ in range(30)]

        self.assertEqual(a_weathers, b_weathers)

    def test_agent_weather_field_tracks_the_last_event(self):
        env = EnvironmentAgent(seed=5)
        event = env.tick()
        self.assertEqual(env.weather, event.weather)

    def test_time_of_day_advances_one_step_per_tick(self):
        env = EnvironmentAgent(seed=5)
        self.assertEqual(env.tick().time_of_day, TimeOfDay.AFTERNOON)
        self.assertEqual(env.tick().time_of_day, TimeOfDay.EVENING)
        self.assertEqual(env.tick().time_of_day, TimeOfDay.NIGHT)
        self.assertEqual(env.tick().time_of_day, TimeOfDay.MORNING)


if __name__ == "__main__":
    unittest.main()
