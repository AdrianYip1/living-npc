"""Travelers as live LLM agents: admitted at a map edge, told about an exit
point on the opposite edge, and moving only via their own move_to calls.
"""
from __future__ import annotations

import json
import math
import re
import tempfile
import unittest
from pathlib import Path

from environment_agent.agent import EnvironmentAgent

from game_agents.agent import TurnResult
from game_agents.identity import Identity
from game_agents.llm import SPEAK_TOOL_NAME, LLMResult, MockLLMClient, ToolCall
from game_agents.registry import NPCRegistry
from game_agents.storage import save_identities
from game_agents.world import HESITATE_SPEED_FACTOR, MAP_MAX, MAP_MIN, NPC_MAX_SPEED, PERSONAL_SPACE, distance

from mini_map.simulation import Simulation

_COORDS = re.compile(r"\((-?\d+), (-?\d+)\)")


class _TravelerLLM:
    """Plays a traveler the way the prompt asks: on arrival, walk to the
    exit point named in the stimulus; on noticing someone, walk over to
    them (interrupting the walk). Everything else just speaks. Records
    every stimulus a traveler received.
    """

    def __init__(self, *, chase_noticed: bool = True):
        self.chase_noticed = chase_noticed
        self.stimuli: list[str] = []

    def complete(self, *, system, messages, tools):
        stimulus = messages[-1]["content"]
        tool_names = {t["name"] for t in tools}
        if "create_identity" in tool_names:
            return MockLLMClient().complete(system=system, messages=messages, tools=tools)
        if any("few words" in t["description"] for t in tools if t["name"] == SPEAK_TOOL_NAME):
            self.stimuli.append(stimulus)
        if stimulus.startswith("You've just arrived"):
            x, y = _COORDS.findall(stimulus)[-1]  # the exit point
            return LLMResult(tool_call=ToolCall(name="move_to", arguments={"x": int(x), "y": int(y)}))
        if stimulus.startswith("You notice") and self.chase_noticed:
            x, y = _COORDS.findall(stimulus)[0]
            return LLMResult(tool_call=ToolCall(name="move_to", arguments={"x": int(x), "y": int(y)}))
        return LLMResult(tool_call=ToolCall(name=SPEAK_TOOL_NAME, arguments={"text": "hm"}))


def _sim(tmp, llm, *, residents=(), places=(), **env_kwargs) -> Simulation:
    npcs_path = Path(tmp) / "npcs.json"
    save_identities(
        [Identity(name=n, traits=[], backstory="", speech_style="", home=home) for n, home in residents], npcs_path
    )
    world_path = Path(tmp) / "world.json"
    world_path.write_text(json.dumps({"places": list(places)}), encoding="utf-8")
    registry = NPCRegistry(npcs_path, Path(tmp) / "memory", llm, world_path=world_path)
    env = EnvironmentAgent(**{"travelers_per_day": (1, 1), "start_minute": 0, "minutes_per_tick": 1440, "seed": 1, **env_kwargs})
    return Simulation(registry, env, seed=3)


def _admit_one(sim: Simulation):
    # Make the next tick deliver exactly one arrival, whatever the tick size.
    env = sim._environment
    env._planned_arrivals = [env.elapsed_minutes + env.minutes_per_tick]
    sim._advance_environment()
    sim.wait_for_pending()
    (traveler,) = sim._registry.travelers()
    return traveler, sim._traveler_state[traveler.identity.name]


def _walk(sim: Simulation, seconds: float) -> None:
    for _ in range(int(seconds * 30)):
        sim._registry.step_movement(1 / 30, hesitating=set(sim._turns_in_flight))
        sim._update_travelers()
        sim.wait_for_pending()


def _on_edge(point) -> bool:
    return point[0] in (MAP_MIN, MAP_MAX) or point[1] in (MAP_MIN, MAP_MAX)


class TravelerAgentTests(unittest.TestCase):
    def test_traveler_arrives_on_an_edge_with_an_exit_on_the_opposite_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = _sim(tmp, _TravelerLLM())
            traveler, st = _admit_one(sim)

            entry = traveler.position  # hasn't taken a step yet
            self.assertTrue(_on_edge(entry))
            self.assertTrue(_on_edge(st.exit_point))
            # Opposite edges: one axis spans the whole map.
            spans = {abs(entry[0] - st.exit_point[0]), abs(entry[1] - st.exit_point[1])}
            self.assertIn(MAP_MAX - MAP_MIN, spans)
            self.assertTrue(sim.state()["traveler_arrivals"][0]["identity"]["name"])

    def test_arrival_turn_names_the_exit_and_the_traveler_walks_there_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _TravelerLLM()
            sim = _sim(tmp, llm)
            traveler, st = _admit_one(sim)

            self.assertIn(f"({st.exit_point[0]}, {st.exit_point[1]})", llm.stimuli[0])
            self.assertEqual(traveler.destination, st.exit_point)
            self.assertIn(f"Your exit point: ({st.exit_point[0]}, {st.exit_point[1]})", traveler.standing_context)

    def test_traveler_leaves_town_on_reaching_its_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = _sim(tmp, _TravelerLLM())
            traveler, _ = _admit_one(sim)
            name = traveler.identity.name

            _walk(sim, 15)  # the map is 200 units wide; NPC top speed is ~43/s

            self.assertIsNone(sim._registry.get(name))
            self.assertNotIn(name, [npc["name"] for npc in sim.state()["npcs"]])

    def test_noticing_someone_can_interrupt_the_walk(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _TravelerLLM()
            sim = _sim(tmp, llm, residents=[("Mara", (0, 0))])
            traveler, st = _admit_one(sim)
            mara = sim._registry.get("Mara")
            # Halfway along the traveler's straight route to its exit.
            mara.position = (
                (traveler.position[0] + st.exit_point[0]) / 2,
                (traveler.position[1] + st.exit_point[1]) / 2,
            )

            _walk(sim, 10)

            self.assertTrue(any(s.startswith("You notice someone nearby: Mara") for s in llm.stimuli))
            # It went over to Mara instead of leaving: still in town, next
            # to her (stopping short of her spot -- see keep_personal_space).
            self.assertIsNotNone(sim._registry.get(traveler.identity.name))
            self.assertLessEqual(distance(traveler.position, mara.position), PERSONAL_SPACE + 1)

    def test_each_person_is_noticed_only_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _TravelerLLM(chase_noticed=False)
            sim = _sim(tmp, llm, residents=[("Mara", (0, 0))])
            traveler, st = _admit_one(sim)
            sim._registry.get("Mara").position = traveler.position

            _walk(sim, 2)

            self.assertEqual(sum(s.startswith("You notice") for s in llm.stimuli), 1)

    def test_traveler_slows_while_its_turn_is_being_decided(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = _sim(tmp, _TravelerLLM())
            traveler, _ = _admit_one(sim)
            _walk(sim, 1)
            self.assertGreater(abs(traveler.velocity[0]) + abs(traveler.velocity[1]), 0)

            sim._turns_in_flight.add(traveler.identity.name)  # a turn is "in flight"
            for _ in range(60):
                sim._registry.step_movement(1 / 30, hesitating=set(sim._turns_in_flight))
            # Hesitates -- slows right down -- rather than freezing mid-route.
            speed = math.hypot(*traveler.velocity)
            self.assertGreater(speed, 0)
            self.assertLessEqual(speed, NPC_MAX_SPEED * HESITATE_SPEED_FACTOR + 1e-6)

    def _refused_by_busy(self, sim, traveler, target):
        sim._record_turn(
            traveler,
            TurnResult(
                utterance=None,
                action={
                    "name": "initiate_conversation",
                    "arguments": {"target_name": target},
                    "result": f"{target} is busy right now.",
                },
            ),
        )

    def test_a_refused_traveler_is_told_when_its_target_is_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _TravelerLLM()
            sim = _sim(tmp, llm, residents=[("Mara", (0, 0)), ("Finn", (0, 5))])
            traveler, st = _admit_one(sim)
            traveler.destination = None
            sim._registry.try_occupy_pair("Mara", "Finn")
            self._refused_by_busy(sim, traveler, "Mara")
            self.assertEqual(st.waiting_for, "Mara")

            sim._update_travelers()
            sim.wait_for_pending()
            self.assertFalse(any("free to talk now" in s for s in llm.stimuli))

            sim._registry.release_pair("Mara", "Finn")
            sim._update_travelers()
            sim.wait_for_pending()
            self.assertTrue(any(s.startswith("Mara has finished their conversation") for s in llm.stimuli))
            self.assertIsNone(st.waiting_for)

    def test_a_traveler_gives_up_and_leaves_after_waiting_ten_minutes(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = _sim(tmp, _TravelerLLM(), residents=[("Mara", (0, 0)), ("Finn", (0, 5))], minutes_per_tick=1)
            traveler, st = _admit_one(sim)
            traveler.destination = None
            sim._registry.try_occupy_pair("Mara", "Finn")
            self._refused_by_busy(sim, traveler, "Mara")

            for _ in range(Simulation.TRAVELER_MAX_WAIT_MINUTES - 1):
                sim._environment.tick()
            sim._update_travelers()
            self.assertEqual(st.waiting_for, "Mara")  # 9 minutes: still waiting
            self.assertFalse(st.leaving)

            sim._environment.tick()
            sim._update_travelers()
            self.assertTrue(st.leaving)
            self.assertEqual(traveler.destination, st.exit_point)
            self.assertIn("tired of waiting for Mara", sim.state()["npcs"][-1]["activity"])

            # Leaving is for good: noticing someone on the way doesn't pull it back.
            sim._registry.release_pair("Mara", "Finn")
            turns_before = len(sim._turns_in_flight)
            sim._update_travelers()
            self.assertEqual(traveler.destination, st.exit_point)
            self.assertEqual(len(sim._turns_in_flight), turns_before)

    def test_state_says_when_a_traveler_is_stopped_deciding(self):
        # The page has to brake along with the server, or it predicts the
        # walk on and snaps the icon back on every poll (a visible jiggle).
        with tempfile.TemporaryDirectory() as tmp:
            sim = _sim(tmp, _TravelerLLM())
            traveler, _ = _admit_one(sim)
            name = traveler.identity.name

            def deciding():
                return next(n for n in sim.state()["npcs"] if n["name"] == name)["deciding"]

            self.assertFalse(deciding())
            sim._turns_in_flight.add(name)
            self.assertTrue(deciding())

    def test_a_traveler_that_never_moves_is_sent_out_after_the_max_stay(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = _sim(tmp, MockLLMClient(), minutes_per_tick=60)  # mock only ever speaks
            traveler, st = _admit_one(sim)
            self.assertIsNone(traveler.destination)

            for _ in range(Simulation.TRAVELER_MAX_STAY_MINUTES // 60):
                sim._environment.tick()
            sim._update_travelers()

            self.assertEqual(traveler.destination, st.exit_point)

    def test_reaching_a_waypoint_prompts_what_next(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _TravelerLLM()
            sim = _sim(tmp, llm)
            traveler, st = _admit_one(sim)
            traveler.destination = (0, 0)  # detoured into town

            _walk(sim, 6)

            reached = [s for s in llm.stimuli if s.startswith("You've reached (0, 0)")]
            self.assertEqual(len(reached), 1)
            self.assertIn(f"({st.exit_point[0]}, {st.exit_point[1]})", reached[0])

    def test_reaching_someone_says_who_is_close_enough_to_talk_to(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _TravelerLLM(chase_noticed=False)
            sim = _sim(tmp, llm, residents=[("Mara", (3, 0)), ("Finn", (80, 80))])
            traveler, _ = _admit_one(sim)
            traveler.destination = (0, 0)

            _walk(sim, 6)

            (reached,) = [s for s in llm.stimuli if s.startswith("You've reached (0, 0)")]
            self.assertIn("Close enough to talk to: Mara.", reached)
            self.assertNotIn("Finn", reached)

    def test_idle_traveler_is_prompted_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _TravelerLLM()
            sim = _sim(tmp, llm, minutes_per_tick=1)
            traveler, _ = _admit_one(sim)
            traveler.destination = None  # it "stopped" somewhere
            traveler.velocity = (0.0, 0.0)
            before = len(llm.stimuli)

            for _ in range(Simulation.TRAVELER_IDLE_TURN_MINUTES):
                sim._environment.tick()
            sim._update_travelers()
            sim.wait_for_pending()

            self.assertEqual(len(llm.stimuli), before + 1)
            self.assertTrue(llm.stimuli[-1].startswith("You're standing at"))

    def test_resident_query_round_skips_travelers(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _TravelerLLM()
            sim = _sim(tmp, llm, residents=[("Mara", (0, 0))])
            _admit_one(sim)
            before = len(llm.stimuli)
            sim._query_npcs()
            self.assertEqual(len(llm.stimuli), before)

    def test_state_flags_travelers(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = _sim(tmp, _TravelerLLM(), residents=[("Mara", (0, 0))])
            traveler, _ = _admit_one(sim)
            flags = {npc["name"]: npc["traveler"] for npc in sim.state()["npcs"]}
            self.assertEqual(flags, {"Mara": False, traveler.identity.name: True})

    def test_player_can_talk_to_a_traveler_and_it_answers_briefly(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = _sim(tmp, _TravelerLLM())
            traveler, _ = _admit_one(sim)
            name = traveler.identity.name

            self.assertTrue(sim.start_conversation(name))
            reply = sim.say(name, "Where are you headed?")
            sim.end_conversation(name)

            self.assertEqual(reply["utterance"], "hm")


if __name__ == "__main__":
    unittest.main()
