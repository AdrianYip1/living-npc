from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
