"""Why travelers stop in town (mini_map/traveler_purpose.py): rolled on
arrival, threaded into what they're told, and -- with the chatty mock
backend playing them -- actually acted on.
"""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from environment_agent.agent import EnvironmentAgent

from game_agents.chatty_mock import ChattyMockLLMClient
from game_agents.identity import Identity
from game_agents.llm import MockLLMClient
from game_agents.registry import NPCRegistry
from game_agents.storage import save_identities

from mini_map.simulation import Simulation
from mini_map.traveler_purpose import BUY, PASSING_THROUGH, SOCIALIZE, TravelerPurpose, pick_purpose

_FORGE = {"name": "The Forge", "position": [12, -30], "description": "Hot and loud."}
_DOCKS = {"name": "The Docks", "position": [70, 45], "description": "Wet and windy."}
_MARA = Identity(
    name="Mara",
    traits=[],
    backstory="",
    speech_style="",
    home=(12, -30),
    workplace=(12, -30),
    starting_items={"horseshoe": 4, "hammer": 1},
)
_FINN = Identity(name="Finn", traits=[], backstory="", speech_style="", home=(70, 45), workplace=(70, 45))


def _sim(tmp, llm, *, residents=(_MARA,), places=(_FORGE, _DOCKS)) -> Simulation:
    npcs_path = Path(tmp) / "npcs.json"
    save_identities(list(residents), npcs_path)
    world_path = Path(tmp) / "world.json"
    world_path.write_text(json.dumps({"places": list(places)}), encoding="utf-8")
    registry = NPCRegistry(npcs_path, Path(tmp) / "memory", llm, world_path=world_path)
    env = EnvironmentAgent(travelers_per_day=(1, 1), start_minute=8 * 60, minutes_per_tick=1, seed=1)
    return Simulation(registry, env, seed=3, line_pacing=False)


def _admit(sim: Simulation, purpose: TravelerPurpose):
    env = sim._environment
    env._planned_arrivals = [env.elapsed_minutes + env.minutes_per_tick]
    with mock.patch("mini_map.simulation.pick_purpose", return_value=purpose):
        sim._advance_environment()
        sim.wait_for_pending()
    (traveler,) = sim._registry.travelers()
    return traveler, sim._traveler_state[traveler.identity.name]


def _run(sim: Simulation, seconds: float) -> None:
    """Real-time-ish: movement at 30Hz, the clock at a game-minute per
    real second, and every background turn/conversation settled per step.
    """
    for step in range(int(seconds * 30)):
        sim._registry.step_movement(1 / 30, hesitating=set(sim._turns_in_flight))
        if step % 30 == 0:
            sim._environment.tick()
        sim._update_travelers()
        sim.wait_for_pending()


class PickPurposeTests(unittest.TestCase):
    def _residents(self, tmp):
        return _sim(tmp, MockLLMClient(), residents=(_MARA, _FINN))._registry.residents()

    def test_all_three_purposes_come_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            residents = self._residents(tmp)
            rng = random.Random(0)
            kinds = Counter(pick_purpose(rng, [_FORGE, _DOCKS], residents).kind for _ in range(200))
            self.assertEqual(set(kinds), {PASSING_THROUGH, SOCIALIZE, BUY})

    def test_a_buyer_wants_something_its_seller_has_to_spare(self):
        with tempfile.TemporaryDirectory() as tmp:
            residents = self._residents(tmp)
            rng = random.Random(0)
            buys = [p for p in (pick_purpose(rng, [_FORGE], residents) for _ in range(100)) if p.kind == BUY]
            self.assertTrue(buys)
            self.assertTrue(all((p.seller, p.item, p.place) == ("Mara", "horseshoe", _FORGE) for p in buys))

    def test_a_place_with_nothing_to_sell_is_visited_instead(self):
        with tempfile.TemporaryDirectory() as tmp:
            residents = self._residents(tmp)  # Finn at the docks carries nothing
            rng = random.Random(0)
            kinds = {pick_purpose(rng, [_DOCKS], residents).kind for _ in range(100)}
            self.assertEqual(kinds, {PASSING_THROUGH, SOCIALIZE})

    def test_no_places_means_passing_through(self):
        rng = random.Random(0)
        self.assertEqual({pick_purpose(rng, [], []).kind for _ in range(30)}, {PASSING_THROUGH})


class AdmittedPurposeTests(unittest.TestCase):
    def test_the_purpose_is_in_the_standing_prompt_and_the_arrival_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            purpose = TravelerPurpose(BUY, place=_FORGE, item="horseshoe", seller="Mara")
            sim = _sim(tmp, MockLLMClient())
            traveler, st = _admit(sim, purpose)

            self.assertIn(
                "buy horseshoe from Mara, who works at The Forge at (12, -30)", traveler.standing_context
            )
            self.assertIn("buy horseshoe from Mara", traveler.identity.goals[0])
            self.assertGreaterEqual(traveler.inventory.money, Simulation.TRAVELER_BUYER_MIN_MONEY)
            self.assertEqual(st.reminder(traveler, sim._environment.elapsed_minutes), " You still want to buy horseshoe from Mara at The Forge at (12, -30).")

    def test_passing_through_mentions_no_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            traveler, st = _admit(_sim(tmp, MockLLMClient()), TravelerPurpose(PASSING_THROUGH))
            self.assertIn("just passing through", traveler.standing_context)
            self.assertNotIn("spend some time at", traveler.standing_context)
            self.assertNotIn("to buy", traveler.standing_context)

    def test_reaching_the_place_counts_as_arriving(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = _sim(tmp, MockLLMClient())
            traveler, st = _admit(sim, TravelerPurpose(SOCIALIZE, place=_DOCKS))
            self.assertIn("still want to spend some time at The Docks", st.reminder(traveler, sim._environment.elapsed_minutes))
            traveler.position, traveler.destination = (70.0, 40.0), None

            sim._update_travelers()

            self.assertTrue(st.visited)
            self.assertIn("You're spending time at The Docks", st.reminder(traveler, sim._environment.elapsed_minutes))

    def test_the_reminder_notices_the_purchase(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = _sim(tmp, MockLLMClient())
            traveler, st = _admit(sim, TravelerPurpose(BUY, place=_FORGE, item="horseshoe", seller="Mara"))
            traveler.position = (12.0, -25.0)
            sim._registry.try_occupy_pair(traveler.identity.name, "Mara")  # trades only happen mid-conversation

            traveler.tools.execute("buy_item", {"item": "horseshoe", "seller_name": "Mara", "total_price": 5})

            self.assertEqual(st.reminder(traveler, sim._environment.elapsed_minutes), " You've bought the horseshoe you came for.")


class TimeInTownTests(unittest.TestCase):
    def test_the_reminder_says_how_long_the_traveler_has_been_in_town(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = _sim(tmp, MockLLMClient())
            traveler, st = _admit(sim, TravelerPurpose(SOCIALIZE, place=_DOCKS))
            arrived = st.arrived_minute

            self.assertNotIn("in town", st.reminder(traveler, arrived + 10))
            self.assertTrue(st.reminder(traveler, arrived + 45).endswith(" You've been in town 45 minutes so far."))
            self.assertTrue(st.reminder(traveler, arrived + 60).endswith(" You've been in town 1 hour so far."))
            self.assertTrue(st.reminder(traveler, arrived + 150).endswith(" You've been in town 2 hours 30 minutes so far."))


class PurposeBehaviorTests(unittest.TestCase):
    """The chatty mock playing each purpose through the real simulation."""

    def test_a_buyer_goes_to_the_seller_and_buys_the_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = _sim(tmp, ChattyMockLLMClient(seed=2))
            traveler, st = _admit(sim, TravelerPurpose(BUY, place=_FORGE, item="horseshoe", seller="Mara"))

            _run(sim, 12)

            speech = sim.state()["speech"]
            trades = [line["text"] for line in speech if line["kind"] == "trade"]
            self.assertEqual(len(trades), 1)
            self.assertRegex(trades[0], rf"^{traveler.identity.name} bought 1 x horseshoe from Mara for \d+ coins\.$")
            self.assertTrue(any("I'm after a horseshoe" in line["text"] for line in speech))
            self.assertEqual(sim._registry.get("Mara").inventory.count("horseshoe"), 3)

    def test_a_visitor_goes_to_its_place_and_talks_to_whoever_is_there(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = _sim(tmp, ChattyMockLLMClient(seed=2))
            traveler, st = _admit(sim, TravelerPurpose(SOCIALIZE, place=_FORGE))

            _run(sim, 12)

            self.assertTrue(st.visited)
            speakers = {line["speaker"] for line in sim.state()["speech"] if line["listener"]}
            self.assertEqual(speakers, {traveler.identity.name, "Mara"})

    def test_someone_passing_through_heads_straight_for_the_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = _sim(tmp, ChattyMockLLMClient(seed=2))
            traveler, st = _admit(sim, TravelerPurpose(PASSING_THROUGH))

            self.assertEqual(traveler.destination, st.exit_point)


if __name__ == "__main__":
    unittest.main()
