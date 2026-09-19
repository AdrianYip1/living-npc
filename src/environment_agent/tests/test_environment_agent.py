from __future__ import annotations

import random
import unittest
from unittest import mock

from environment_agent.agent import EnvironmentAgent
from environment_agent.temperature import MAX_TEMPERATURE, MIN_TEMPERATURE, next_temperature
from environment_agent.time_of_day import TimeOfDay, format_clock, phase_for_minute
from environment_agent.travelers import _FIRST_NAMES, _ORIGINS, _REASONS, _TRAITS, invent_traveler, spare_names
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

        self.assertIn((traveler.name, traveler.gender), _FIRST_NAMES)
        self.assertIn(traveler.origin, _ORIGINS)
        self.assertIn(traveler.reason, _REASONS)
        self.assertEqual(len(traveler.traits), 2)
        self.assertTrue(set(traveler.traits).issubset(set(_TRAITS)))

    def test_traits_are_not_duplicated(self):
        traveler = invent_traveler(random.Random(7))
        self.assertEqual(len(set(traveler.traits)), len(traveler.traits))

    def test_spare_names_match_the_gender_and_are_unique(self):
        for gender in ("male", "female"):
            names = spare_names(gender)
            self.assertTrue(names)
            self.assertTrue(all((name, gender) in _FIRST_NAMES for name in names))
        pool = [name.lower() for name, _ in _FIRST_NAMES]
        self.assertEqual(len(pool), len(set(pool)))
        self.assertEqual(len(spare_names("")), len(_FIRST_NAMES))  # unknown gender: everyone


class NextTemperatureTests(unittest.TestCase):
    def test_stays_within_bounds_across_every_combination(self):
        rng = random.Random(1)
        for _ in range(50):
            for time_of_day in TimeOfDay:
                for weather in Weather:
                    value = next_temperature(time_of_day, weather, rng)
                    self.assertGreaterEqual(value, MIN_TEMPERATURE)
                    self.assertLessEqual(value, MAX_TEMPERATURE)

    def test_same_seed_is_reproducible(self):
        a = next_temperature(TimeOfDay.AFTERNOON, Weather.CLEAR, random.Random(42))
        b = next_temperature(TimeOfDay.AFTERNOON, Weather.CLEAR, random.Random(42))
        self.assertEqual(a, b)

    def test_stormy_afternoon_is_colder_than_clear_afternoon(self):
        # Jitter is only +/-2, so this holds regardless of the rng draw.
        clear = next_temperature(TimeOfDay.AFTERNOON, Weather.CLEAR, random.Random(1))
        stormy = next_temperature(TimeOfDay.AFTERNOON, Weather.STORMY, random.Random(1))
        self.assertLess(stormy, clear)

    def test_afternoon_is_warmer_than_night_for_the_same_weather(self):
        night = next_temperature(TimeOfDay.NIGHT, Weather.CLOUDY, random.Random(1))
        afternoon = next_temperature(TimeOfDay.AFTERNOON, Weather.CLOUDY, random.Random(1))
        self.assertLess(night, afternoon)


class PhaseForMinuteTests(unittest.TestCase):
    def test_each_quarter_of_the_day_maps_to_its_phase(self):
        self.assertEqual(phase_for_minute(0), TimeOfDay.NIGHT)
        self.assertEqual(phase_for_minute(359), TimeOfDay.NIGHT)
        self.assertEqual(phase_for_minute(360), TimeOfDay.MORNING)  # 06:00
        self.assertEqual(phase_for_minute(719), TimeOfDay.MORNING)
        self.assertEqual(phase_for_minute(720), TimeOfDay.AFTERNOON)  # 12:00
        self.assertEqual(phase_for_minute(1079), TimeOfDay.AFTERNOON)
        self.assertEqual(phase_for_minute(1080), TimeOfDay.EVENING)  # 18:00
        self.assertEqual(phase_for_minute(1439), TimeOfDay.EVENING)

    def test_wraps_for_values_outside_a_single_day(self):
        self.assertEqual(phase_for_minute(1440), TimeOfDay.NIGHT)  # exactly one day past
        self.assertEqual(phase_for_minute(1440 + 360), TimeOfDay.MORNING)


class FormatClockTests(unittest.TestCase):
    def test_formats_as_zero_padded_hh_mm(self):
        self.assertEqual(format_clock(0), "00:00")
        self.assertEqual(format_clock(75), "01:15")
        self.assertEqual(format_clock(1439), "23:59")

    def test_wraps_for_values_outside_a_single_day(self):
        self.assertEqual(format_clock(1440), "00:00")


class EnvironmentAgentTickTests(unittest.TestCase):
    """tick() is the one-shot, no-lookahead entry point standing in for a
    future LLM turn -- these check the deterministic internals behave, not
    that any particular weather or traveler was chosen.
    """

    def test_zero_travelers_per_day_never_spawns_anyone(self):
        env = EnvironmentAgent(travelers_per_day=(0, 0), minutes_per_tick=60, seed=1)
        for _ in range(24 * 5):  # five full days
            self.assertEqual(env.tick().travelers_arrived, [])

    def test_each_full_day_spawns_a_count_within_the_range(self):
        # Start at midnight so every day is planned in full.
        env = EnvironmentAgent(travelers_per_day=(3, 5), start_minute=0, minutes_per_tick=1, seed=4)
        for _ in range(10):
            arrived = sum(len(env.tick().travelers_arrived) for _ in range(1440))
            self.assertGreaterEqual(arrived, 3)
            self.assertLessEqual(arrived, 5)

    def test_fixed_count_spawns_exactly_that_many_per_day(self):
        env = EnvironmentAgent(travelers_per_day=(4, 4), start_minute=0, minutes_per_tick=1, seed=2)
        for _ in range(5):
            self.assertEqual(sum(len(env.tick().travelers_arrived) for _ in range(1440)), 4)

    def test_arrivals_are_spread_across_the_day_not_all_at_once(self):
        env = EnvironmentAgent(travelers_per_day=(6, 6), start_minute=0, minutes_per_tick=1, seed=9)
        arrival_ticks = [i for i in range(1440) if env.tick().travelers_arrived]
        self.assertGreater(len(arrival_ticks), 1)

    def test_big_ticks_that_skip_whole_days_still_deliver_every_arrival(self):
        env = EnvironmentAgent(travelers_per_day=(2, 2), start_minute=0, minutes_per_tick=1440 * 3, seed=3)
        # One tick crosses three midnights: day 0's two arrivals plus days 1-2's.
        self.assertEqual(len(env.tick().travelers_arrived), 6)

    def test_starting_day_only_keeps_arrivals_after_the_start_time(self):
        env = EnvironmentAgent(travelers_per_day=(10, 10), start_minute=23 * 60 + 59, seed=1)
        # At 23:59 almost the whole day has passed, so nearly all are dropped.
        self.assertLessEqual(len(env.planned_arrivals_today), 1)

    def test_same_seed_plans_the_same_arrival_times(self):
        a = EnvironmentAgent(seed=11, start_minute=0)
        b = EnvironmentAgent(seed=11, start_minute=0)
        self.assertEqual(a.planned_arrivals_today, b.planned_arrivals_today)

    def test_invalid_travelers_per_day_is_rejected(self):
        with self.assertRaises(ValueError):
            EnvironmentAgent(travelers_per_day=(5, 2))

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

    def test_time_of_day_derived_from_the_advancing_clock(self):
        # One tick per 6-hour quarter -- crosses exactly one phase boundary
        # each time, mirroring the old fixed-phase-per-tick behavior.
        env = EnvironmentAgent(start_minute=0, minutes_per_tick=360, seed=5)
        self.assertEqual(env.tick().time_of_day, TimeOfDay.MORNING)
        self.assertEqual(env.tick().time_of_day, TimeOfDay.AFTERNOON)
        self.assertEqual(env.tick().time_of_day, TimeOfDay.EVENING)
        self.assertEqual(env.tick().time_of_day, TimeOfDay.NIGHT)

    def test_clock_advances_by_minutes_per_tick_and_wraps_at_midnight(self):
        env = EnvironmentAgent(start_minute=23 * 60 + 50, minutes_per_tick=15, seed=5)  # 23:50
        self.assertEqual(env.tick().clock, "00:05")

    def test_clock_starts_at_the_configured_minute(self):
        env = EnvironmentAgent(start_minute=8 * 60, seed=5)
        self.assertEqual(env.clock, "08:00")
        self.assertEqual(env.time_of_day, TimeOfDay.MORNING)

    def test_default_minutes_per_tick_is_one_for_a_smooth_clock(self):
        # Regression guard: the clock must always count up in its smallest
        # visible step by default, regardless of tick_interval_s -- a
        # bigger default here would mean the clock visibly jumps rather
        # than ticks, no matter how the real-time cadence is configured.
        env = EnvironmentAgent(start_minute=8 * 60, seed=5)
        self.assertEqual(env.tick().clock, "08:01")

    def test_weather_and_temperature_stay_fixed_within_the_same_hour(self):
        env = EnvironmentAgent(start_minute=0, minutes_per_tick=1, seed=7)
        initial_weather = env.weather
        initial_temperature = env.temperature
        for _ in range(59):  # 00:01 through 00:59 -- still hour 0
            event = env.tick()
            self.assertEqual(event.weather, initial_weather)
            self.assertFalse(event.weather_changed)
            self.assertEqual(event.temperature, initial_temperature)

    def test_weather_does_not_reroll_within_the_same_hour(self):
        env = EnvironmentAgent(start_minute=0, minutes_per_tick=1, seed=7, travelers_per_day=(0, 0))
        with mock.patch("environment_agent.agent.next_weather") as mocked_next_weather:
            env.tick()  # 00:00 -> 00:01, still hour 0
            mocked_next_weather.assert_not_called()

    def test_weather_rerolls_exactly_when_the_hour_changes(self):
        env = EnvironmentAgent(start_minute=59, minutes_per_tick=1, seed=7, travelers_per_day=(0, 0))
        with mock.patch("environment_agent.agent.next_weather", return_value=env.weather) as mocked_next_weather:
            env.tick()  # 00:59 -> 01:00, crosses into hour 1
            mocked_next_weather.assert_called_once()

    def test_weather_reroll_happens_once_even_if_a_tick_skips_multiple_hours(self):
        env = EnvironmentAgent(start_minute=0, minutes_per_tick=180, seed=7, travelers_per_day=(0, 0))  # 3 hours/tick
        with mock.patch("environment_agent.agent.next_weather", return_value=env.weather) as mocked_next_weather:
            env.tick()  # 00:00 -> 03:00, three hour boundaries crossed at once
            mocked_next_weather.assert_called_once()

    def test_temperature_is_set_on_construction_and_stays_in_bounds(self):
        env = EnvironmentAgent(seed=5)
        self.assertGreaterEqual(env.temperature, MIN_TEMPERATURE)
        self.assertLessEqual(env.temperature, MAX_TEMPERATURE)

    def test_event_temperature_matches_agent_state_after_tick(self):
        env = EnvironmentAgent(seed=5)
        event = env.tick()
        self.assertEqual(event.temperature, env.temperature)
        self.assertGreaterEqual(event.temperature, MIN_TEMPERATURE)
        self.assertLessEqual(event.temperature, MAX_TEMPERATURE)


if __name__ == "__main__":
    unittest.main()
