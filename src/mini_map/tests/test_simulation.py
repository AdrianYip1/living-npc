from __future__ import annotations

import tempfile
import threading
import time
import unittest
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


def _registry(tmp, llm, *, names=("Mara", "Finn"), **kwargs) -> NPCRegistry:
    npcs_path = Path(tmp) / "npcs.json"
    save_identities([_identity(n) for n in names], npcs_path)
    return NPCRegistry(npcs_path, Path(tmp) / "memory", llm, **kwargs)


class RunForeverThreadingTests(unittest.TestCase):
    """run_forever() itself: confirms the environment and NPC-query loops
    are genuinely independent threads -- a slow/blocked NPC query must not
    stall the environment clock, which is the whole reason tick_interval_s
    and npc_query_interval_s were split out of one shared loop.
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

            sim = Simulation(registry, environment, tick_interval_s=0.02, npc_query_interval_s=100)
            with mock.patch.object(Simulation, "_query_npcs", blocking_query_npcs):
                thread = threading.Thread(target=sim.run_forever, daemon=True)
                thread.start()
                try:
                    # npc_query_interval_s=100 means the query loop's first
                    # (blocked) call never returns in this window -- the env
                    # loop ticking many times at 0.02s anyway is what proves
                    # the two loops are independent, not serialized.
                    time.sleep(0.3)
                    self.assertGreater(len(tick_calls), 3)
                finally:
                    release_npc_query.set()
                    sim.stop()
                    thread.join(timeout=2)


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
            registry = _registry(tmp, MockLLMClient())
            environment = EnvironmentAgent(seed=1)
            initial_minute = environment.minute_of_day
            sim = Simulation(registry, environment, tick_interval_s=0.02, npc_query_interval_s=0.02)
            sim.pause()  # paused before the loops get a chance to run at all

            thread = threading.Thread(target=sim.run_forever, daemon=True)
            thread.start()
            try:
                time.sleep(0.2)
                self.assertEqual(environment.minute_of_day, initial_minute)
                self.assertEqual(registry.get("Mara").memory.all(), [])
                self.assertEqual(registry.get("Finn").memory.all(), [])
            finally:
                sim.stop()
                thread.join(timeout=2)

    def test_run_forever_resumes_advancing_after_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            environment = EnvironmentAgent(seed=1)
            sim = Simulation(registry, environment, tick_interval_s=0.02, npc_query_interval_s=0.02)
            sim.pause()

            thread = threading.Thread(target=sim.run_forever, daemon=True)
            thread.start()
            try:
                time.sleep(0.1)
                sim.resume()
                time.sleep(0.2)
                # Several rounds could fire in this window at a 0.02s
                # interval -- the point is just that it's no longer zero.
                self.assertGreaterEqual(len(registry.get("Mara").memory.all()), 1)
            finally:
                sim.stop()
                thread.join(timeout=2)


class SimulationTickTests(unittest.TestCase):
    def test_tick_stimulates_every_idle_npc_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            sim.tick()

            self.assertEqual(len(registry.get("Mara").memory.all()), 1)
            self.assertEqual(len(registry.get("Finn").memory.all()), 1)

    def test_busy_npc_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            registry.try_occupy_pair("Mara", "Finn")
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            sim.tick()

            self.assertEqual(registry.get("Mara").memory.all(), [])
            self.assertEqual(registry.get("Finn").memory.all(), [])

    def test_conversation_target_is_not_independently_ticked_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, _NameAwareLLM(), conversation_turns=2)
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            sim.tick()

            # If the target got ticked twice -- once inside the nested
            # conversation, once again independently by the outer loop --
            # Finn (who always just speaks) would pick up a second memory.
            self.assertEqual(len(registry.get("Finn").memory.all()), 1)

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

    def test_state_reports_the_configured_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            sim = Simulation(registry, EnvironmentAgent(seed=1), tick_interval_s=2.5, npc_query_interval_s=30.0)

            state = sim.state()

            self.assertEqual(state["tick_interval_s"], 2.5)
            self.assertEqual(state["npc_query_interval_s"], 30.0)

    def test_state_reports_queries_per_game_minute(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            # 1 real second/tick, 1 game-minute/tick (default) -> 1 game-min
            # passes per real second. Querying every 15 real seconds is
            # then one round per 15 game-minutes, i.e. 1/15 per game-minute.
            environment = EnvironmentAgent(seed=1, minutes_per_tick=1)
            sim = Simulation(registry, environment, tick_interval_s=1.0, npc_query_interval_s=15.0)

            self.assertAlmostEqual(sim.state()["queries_per_game_minute"], 1 / 15)

    def test_queries_per_game_minute_accounts_for_minutes_per_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, MockLLMClient())
            # Same real-time rates as above, but each tick now covers 2
            # game-minutes instead of 1 -- game-time passes twice as fast
            # per real second, so the same query cadence covers twice the
            # in-game ground: half as many queries per game-minute.
            environment = EnvironmentAgent(seed=1, minutes_per_tick=2)
            sim = Simulation(registry, environment, tick_interval_s=1.0, npc_query_interval_s=15.0)

            self.assertAlmostEqual(sim.state()["queries_per_game_minute"], 1 / 30)

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
            registry = _registry(tmp, MockLLMClient())
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            self.assertTrue(sim.start_conversation("Mara"))
            self.assertTrue(registry.is_busy("Mara"))

            sim.tick()
            self.assertEqual(registry.get("Mara").memory.all(), [])  # skipped, busy talking to the player
            self.assertEqual(len(registry.get("Finn").memory.all()), 1)  # unaffected

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
            registry = _registry(tmp, MockLLMClient())
            sim = Simulation(registry, EnvironmentAgent(seed=1))
            sim.start_conversation("Mara")

            sim.end_conversation("Mara")

            self.assertFalse(registry.is_busy("Mara"))
            sim.tick()
            self.assertEqual(len(registry.get("Mara").memory.all()), 1)

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
