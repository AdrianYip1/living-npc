from __future__ import annotations

import re
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from environment_agent.agent import EnvironmentAgent

from game_agents.identity import Identity
from game_agents.llm import SPEAK_TOOL_NAME, LLMResult, MockLLMClient, ToolCall
from game_agents.registry import NPCRegistry
from game_agents.storage import save_identities

from mini_map.simulation import Simulation


def _identity(name: str) -> Identity:
    return Identity(name=name, traits=[], backstory="", speech_style="", goals=[])


class _NameAwareLLM:
    """Mara always tries to start a conversation with Finn; anyone else
    (i.e. Finn, if independently ticked on top of that) always just
    speaks -- a single shared LLMClient instance (NPCRegistry only takes
    one), told apart by which NPC's identity block opens the system prompt.
    """

    def complete(self, *, system, messages, tools):
        if system.startswith("You are Mara."):
            return LLMResult(tool_call=ToolCall(name="initiate_conversation", arguments={"target_name": "Finn"}))
        return LLMResult(tool_call=ToolCall(name=SPEAK_TOOL_NAME, arguments={"text": "hello"}))


class _CountingLLM:
    """Wraps another LLM and counts calls per NPC, told apart the same way
    as _NameAwareLLM. Memory can't stand in for "was queried" -- routine
    world-tick turns deliberately leave no memory (see Agent.respond).
    """

    def __init__(self, inner=None):
        self._inner = inner or MockLLMClient()
        self.calls: Counter[str] = Counter()

    def complete(self, *, system, messages, tools):
        match = re.match(r"You are (\w+)\.", system)
        self.calls[match.group(1) if match else ""] += 1
        return self._inner.complete(system=system, messages=messages, tools=tools)


_NPCS_JSON_KEYS = (
    "name", "traits", "backstory", "speech_style", "goals",
    "home", "workplace", "habits", "starting_money", "starting_items",
)


class _IdentityLLM:
    """Answers traveler-identity requests with a fixed identity (optionally
    after blocking on an Event, to simulate a slow backend); anything else
    goes to the mock.
    """

    def __init__(self, name, block_until=None):
        self._name = name
        self._block_until = block_until

    def complete(self, *, system, messages, tools):
        if tools and tools[0]["name"] == "create_identity":
            if self._block_until is not None:
                self._block_until.wait(timeout=5)
            return LLMResult(
                tool_call=ToolCall(
                    name="create_identity",
                    arguments={
                        "name": self._name,
                        "traits": ["wry"],
                        "backstory": "A cartographer mapping the coast road.",
                        "speech_style": "precise",
                        "goals": ["finish the map"],
                        "habits": ["Sketches the town square"],
                        "starting_money": 20,
                        "starting_items": {"map": 1},
                    },
                )
            )
        return MockLLMClient().complete(system=system, messages=messages, tools=tools)


def _registry(tmp, llm, *, names=("Mara", "Finn"), **kwargs) -> NPCRegistry:
    npcs_path = Path(tmp) / "npcs.json"
    save_identities([_identity(n) for n in names], npcs_path)
    return NPCRegistry(npcs_path, Path(tmp) / "memory", llm, **kwargs)


class RunForeverThreadingTests(unittest.TestCase):
    """run_forever() itself: confirms the environment and NPC-query loops
    are genuinely independent threads -- a slow/blocked NPC query must not
    stall the environment clock, which is the whole reason ticks_per_real_
    minute and llm_calls_per_game_hour are paced on separate loops.
    """

    def test_environment_loop_keeps_advancing_while_npc_query_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient(), names=("Mara",))
            environment = EnvironmentAgent(seed=1)

            tick_calls = []
            real_tick = environment.tick

            def counting_tick():
                tick_calls.append(1)
                return real_tick()

            environment.tick = counting_tick

            release_npc_query = threading.Event()
            real_query_npcs = Simulation._query_npcs

            def blocking_query_npcs(self):
                release_npc_query.wait(timeout=2)
                return real_query_npcs(self)

            # ticks_per_real_minute=3000 -> tick_interval_s=0.02.
            # llm_calls_per_game_hour=0.01 -> npc_query_interval_s=120, i.e.
            # the query loop's first (blocked) call never returns in this
            # test's short window.
            sim = Simulation(registry, environment, ticks_per_real_minute=3000, llm_calls_per_game_hour=0.01)
            with mock.patch.object(Simulation, "_query_npcs", blocking_query_npcs):
                thread = threading.Thread(target=sim.run_forever, daemon=True)
                thread.start()
                try:
                    # The env loop ticking many times while the query loop's
                    # blocked call never returns is what proves the two
                    # loops are independent, not serialized.
                    time.sleep(0.3)
                    self.assertGreater(len(tick_calls), 3)
                finally:
                    release_npc_query.set()
                    sim.stop()
                    thread.join(timeout=2)


class TravelerArrivalTests(unittest.TestCase):
    """The page spawns travelers only from state()'s traveler_arrivals, so
    every arrival the environment agent reports has to land there once,
    with a unique, increasing id.
    """

    def test_every_environment_arrival_shows_up_in_state_with_increasing_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            environment = EnvironmentAgent(travelers_per_day=(3, 3), start_minute=0, seed=1)
            sim = Simulation(registry, environment)

            for _ in range(1440):  # one full in-game day at 1 min/tick
                sim._advance_environment()
            sim.wait_for_pending_travelers(timeout=5)

            arrivals = sim.state()["traveler_arrivals"]
            self.assertEqual([a["id"] for a in arrivals], [1, 2, 3])
            for arrival in arrivals:
                self.assertEqual(set(arrival["identity"]), set(_NPCS_JSON_KEYS))
                self.assertTrue(arrival["identity"]["name"])
                self.assertRegex(arrival["arrived_at"], r"^\d\d:\d\d$")

    def test_llm_generated_identity_is_what_lands_in_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, _IdentityLLM("Selwyn"))
            sim = Simulation(registry, EnvironmentAgent(travelers_per_day=(1, 1), start_minute=0, minutes_per_tick=1440, seed=1))
            sim._advance_environment()
            sim.wait_for_pending_travelers(timeout=5)

            identity = sim.state()["traveler_arrivals"][0]["identity"]
            self.assertEqual(identity["name"], "Selwyn")
            self.assertEqual(identity["starting_items"], {"map": 1})

    def test_slow_identity_generation_does_not_block_the_environment_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            release = threading.Event()
            registry = _registry(tmp, _IdentityLLM("Selwyn", block_until=release))
            environment = EnvironmentAgent(travelers_per_day=(1, 1), start_minute=0, minutes_per_tick=1440, seed=1)
            sim = Simulation(registry, environment)
            try:
                started = time.monotonic()
                sim._advance_environment()  # queues the (blocked) generation
                self.assertLess(time.monotonic() - started, 1.0)
                self.assertEqual(sim.state()["traveler_arrivals"], [])  # not ready yet
            finally:
                release.set()
            sim.wait_for_pending_travelers(timeout=5)
            self.assertEqual(len(sim.state()["traveler_arrivals"]), 1)

    def test_no_arrivals_when_the_environment_sends_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            sim = Simulation(registry, EnvironmentAgent(travelers_per_day=(0, 0), seed=1))
            for _ in range(100):
                sim._advance_environment()
            self.assertEqual(sim.state()["traveler_arrivals"], [])

    def test_only_the_most_recent_arrivals_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            # 1440 min/tick -> 10 arrivals per tick, all in one beat.
            environment = EnvironmentAgent(travelers_per_day=(10, 10), start_minute=0, minutes_per_tick=1440, seed=1)
            sim = Simulation(registry, environment)
            for _ in range(5):
                sim._advance_environment()
            sim.wait_for_pending_travelers(timeout=5)

            ids = [a["id"] for a in sim.state()["traveler_arrivals"]]
            self.assertEqual(len(ids), Simulation.RECENT_TRAVELERS)
            self.assertEqual(ids[-1], 50)


class SimulationPauseTests(unittest.TestCase):
    def test_not_paused_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            sim = Simulation(registry, EnvironmentAgent(seed=1))
            self.assertFalse(sim.is_paused())

    def test_pause_and_resume_toggle_is_paused(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            sim.pause()
            self.assertTrue(sim.is_paused())

            sim.resume()
            self.assertFalse(sim.is_paused())

    def test_state_reports_the_paused_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            sim.pause()

            self.assertTrue(sim.state()["paused"])

    def test_run_forever_does_not_advance_or_query_npcs_while_paused(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _CountingLLM()
            registry = _registry(tmp, llm)
            environment = EnvironmentAgent(seed=1)
            initial_minute = environment.minute_of_day
            # ticks_per_real_minute=3000 -> tick_interval_s=0.02;
            # llm_calls_per_game_hour=60 -> npc_query_interval_s=0.02 too
            # (both fast, for a short test).
            sim = Simulation(registry, environment, ticks_per_real_minute=3000, llm_calls_per_game_hour=60)
            sim.pause()  # paused before the loops get a chance to run at all

            thread = threading.Thread(target=sim.run_forever, daemon=True)
            thread.start()
            try:
                time.sleep(0.2)
                self.assertEqual(environment.minute_of_day, initial_minute)
                self.assertEqual(llm.calls, Counter())
            finally:
                sim.stop()
                thread.join(timeout=2)

    def test_run_forever_resumes_advancing_after_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _CountingLLM()
            registry = _registry(tmp, llm)
            environment = EnvironmentAgent(seed=1)
            sim = Simulation(registry, environment, ticks_per_real_minute=3000, llm_calls_per_game_hour=60)
            sim.pause()

            thread = threading.Thread(target=sim.run_forever, daemon=True)
            thread.start()
            try:
                time.sleep(0.1)
                sim.resume()
                time.sleep(0.2)
                # Several rounds could fire in this window at a 0.02s
                # interval -- the point is just that it's no longer zero.
                self.assertGreaterEqual(llm.calls["Mara"], 1)
            finally:
                sim.stop()
                thread.join(timeout=2)


class SimulationTickTests(unittest.TestCase):
    def test_tick_stimulates_every_idle_npc_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _CountingLLM()
            registry = _registry(tmp, llm)
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            sim.tick()

            self.assertEqual(llm.calls, Counter({"Mara": 1, "Finn": 1}))

    def test_routine_idle_turns_leave_no_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            sim.tick()

            # Mock NPCs just speak on the tick -- routine, so nothing to
            # crowd real interactions out of their top memories.
            self.assertEqual(registry.get("Mara").memory.all(), [])
            self.assertEqual(registry.get("Finn").memory.all(), [])

    def test_busy_npc_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _CountingLLM()
            registry = _registry(tmp, llm)
            registry.try_occupy_pair("Mara", "Finn")
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            sim.tick()

            self.assertEqual(llm.calls, Counter())

    def test_conversation_target_is_not_independently_ticked_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, _NameAwareLLM(), conversation_turns=2)
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            sim.tick()

            # If the target got ticked twice -- once inside the nested
            # conversation, once again independently by the outer loop --
            # Finn (who always just speaks) would pick up a second memory.
            self.assertEqual(len(registry.get("Finn").memory.all()), 1)

    def test_refused_conversation_leaves_the_target_free_to_act(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _CountingLLM(_NameAwareLLM())
            registry = _registry(tmp, llm, conversation_turns=2)
            registry.get("Finn").position = (50, 50)  # out of Mara's range
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            sim.tick()

            # No conversation happened, so Finn still gets his own turn
            # instead of a bogus "talked with Mara".
            self.assertEqual(llm.calls["Finn"], 1)
            finn_state = next(n for n in sim.state()["npcs"] if n["name"] == "Finn")
            self.assertEqual(finn_state["activity"], "said: hello")

    def test_state_reports_time_of_day_weather_and_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient(), names=("Mara",))
            registry.get("Mara").position = (5, -5)
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            sim.tick()
            state = sim.state()

            self.assertIn("time_of_day", state)
            self.assertIn("weather", state)
            [npc] = state["npcs"]
            self.assertEqual(npc["name"], "Mara")
            self.assertEqual((npc["x"], npc["y"]), (5, -5))
            self.assertFalse(npc["busy"])
            self.assertTrue(npc["activity"])

    def test_state_reports_the_configured_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            # minutes_per_tick=1 (default) x ticks_per_real_minute=30 -> 30
            # game-minutes/real-minute is what should actually be reported.
            sim = Simulation(
                registry, EnvironmentAgent(seed=1), ticks_per_real_minute=30.0, llm_calls_per_game_hour=6.0
            )

            state = sim.state()

            self.assertEqual(state["game_minutes_per_real_minute"], 30.0)
            self.assertEqual(state["llm_calls_per_game_hour"], 6.0)

    def test_game_minutes_per_real_minute_accounts_for_minutes_per_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            environment = EnvironmentAgent(seed=1, minutes_per_tick=3)
            sim = Simulation(registry, environment, ticks_per_real_minute=10.0, llm_calls_per_game_hour=4.0)

            self.assertEqual(sim.state()["game_minutes_per_real_minute"], 30.0)

    def test_state_reports_llm_calls_per_real_minute(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            # minutes_per_tick=1 (default) x ticks_per_real_minute=60 -> 60
            # game-minutes per real-minute, i.e. exactly 1 game-hour per
            # real-minute. At 4 calls/game-hour, that's 4 calls/real-minute.
            environment = EnvironmentAgent(seed=1, minutes_per_tick=1)
            sim = Simulation(registry, environment, ticks_per_real_minute=60.0, llm_calls_per_game_hour=4.0)

            self.assertAlmostEqual(sim.state()["llm_calls_per_real_minute"], 4.0)

    def test_llm_calls_per_real_minute_accounts_for_minutes_per_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            # Same rates as above, but each tick now covers 2 game-minutes
            # instead of 1 -- game-time passes twice as fast per real
            # minute, so the same game-hour query rate now covers twice the
            # real-time ground: twice as many calls per real-minute.
            environment = EnvironmentAgent(seed=1, minutes_per_tick=2)
            sim = Simulation(registry, environment, ticks_per_real_minute=60.0, llm_calls_per_game_hour=4.0)

            self.assertAlmostEqual(sim.state()["llm_calls_per_real_minute"], 8.0)

    def test_state_before_any_tick_still_reports_every_npc(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            state = sim.state()

            self.assertEqual({npc["name"] for npc in state["npcs"]}, {"Mara", "Finn"})
            self.assertEqual({npc["activity"] for npc in state["npcs"]}, {""})


class SimulationPlayerConversationTests(unittest.TestCase):
    """start_conversation/end_conversation/say -- what backs the mini-map's
    /api/conversation/* routes.
    """

    def test_start_conversation_claims_busy_and_blocks_the_world_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _CountingLLM()
            registry = _registry(tmp, llm)
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            self.assertTrue(sim.start_conversation("Mara"))
            self.assertTrue(registry.is_busy("Mara"))

            sim.tick()
            self.assertEqual(llm.calls["Mara"], 0)  # skipped, busy talking to the player
            self.assertEqual(llm.calls["Finn"], 1)  # unaffected

    def test_start_conversation_fails_for_an_unknown_npc(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            self.assertFalse(sim.start_conversation("Nobody"))

    def test_start_conversation_fails_when_already_busy(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            registry.try_occupy_pair("Mara", "Finn")
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            self.assertFalse(sim.start_conversation("Mara"))

    def test_end_conversation_frees_the_npc_for_the_next_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _CountingLLM()
            registry = _registry(tmp, llm)
            sim = Simulation(registry, EnvironmentAgent(seed=1))
            sim.start_conversation("Mara")

            sim.end_conversation("Mara")

            self.assertFalse(registry.is_busy("Mara"))
            sim.tick()
            self.assertEqual(llm.calls["Mara"], 1)

    def test_say_returns_the_utterance_and_records_a_player_tagged_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient(), names=("Mara",))
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            result = sim.say("Mara", "hello there")

            self.assertIn("hello there", result["utterance"])
            self.assertIsNone(result["action"])
            [memory] = registry.get("Mara").memory.all()
            self.assertEqual(memory.tags, {"player"})

    def test_say_returns_none_for_an_unknown_npc(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            self.assertIsNone(sim.say("Nobody", "hello"))


if __name__ == "__main__":
    unittest.main()
