from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from game_agents.bootstrap import INSTRUCTIONS_PATH
from game_agents.identity import Identity
from game_agents.llm import SPEAK_TOOL_NAME, LLMResult, MockLLMClient, ToolCall
from game_agents.registry import NPCRegistry
from game_agents.storage import save_identities


def _identity(name: str, **kwargs) -> Identity:
    return Identity(name=name, traits=[], backstory="", speech_style="", goals=[], **kwargs)


class _RecordingLLM:
    """Replies with a fixed tool call and remembers the last prompt."""

    def __init__(self, tool_call: ToolCall | None = None):
        self.tool_call = tool_call or ToolCall(name=SPEAK_TOOL_NAME, arguments={"text": "hi"})
        self.last_system = None
        self.last_tools = None

    def complete(self, *, system, messages, tools):
        self.last_system, self.last_tools = system, tools
        return LLMResult(tool_call=self.tool_call)


def _registry(tmp, llm=None, *, residents=(("Mara", (0, 0)),), **kwargs) -> NPCRegistry:
    npcs_path = Path(tmp) / "npcs.json"
    save_identities([_identity(name, home=home) for name, home in residents], npcs_path)
    return NPCRegistry(npcs_path, Path(tmp) / "memory", llm or MockLLMClient(), **kwargs)


class TravelerLifecycleTests(unittest.TestCase):
    def test_added_traveler_is_live_but_not_a_resident(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp)
            registry.add_traveler(_identity("Wren"), position=(-100, 5))

            self.assertTrue(registry.is_traveler("Wren"))
            self.assertEqual(registry.get("Wren").position, (-100, 5))
            self.assertEqual([a.identity.name for a in registry.travelers()], ["Wren"])
            self.assertEqual([a.identity.name for a in registry.residents()], ["Mara"])
            self.assertEqual(registry.resolve("wren"), "Wren")

    def test_traveler_gets_only_move_converse_buy_and_wait(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp)
            agent = registry.add_traveler(_identity("Wren"), position=(0, 0))
            names = {schema["name"] for schema in agent.tools.schemas()}
            self.assertEqual(names, {"move_to", "initiate_conversation", "buy_item", "wait"})

    def test_duplicate_name_is_rejected_and_unique_name_avoids_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp)
            with self.assertRaises(ValueError):
                registry.add_traveler(_identity("mara"), position=(0, 0))
            self.assertEqual(registry.unique_name("Mara"), "Mara 2")
            self.assertEqual(registry.unique_name("Wren"), "Wren")

    def test_remove_traveler(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp)
            registry.add_traveler(_identity("Wren"), position=(0, 0))

            self.assertTrue(registry.remove_traveler("Wren"))
            self.assertIsNone(registry.get("Wren"))
            self.assertFalse(registry.is_traveler("Wren"))

    def test_residents_cannot_be_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp)
            self.assertFalse(registry.remove_traveler("Mara"))
            self.assertIsNotNone(registry.get("Mara"))

    def test_busy_traveler_cannot_leave(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp)
            registry.add_traveler(_identity("Wren"), position=(0, 0))
            registry.try_occupy("Wren")

            self.assertFalse(registry.remove_traveler("Wren"))
            registry.release("Wren")
            self.assertTrue(registry.remove_traveler("Wren"))

    def test_departed_traveler_cannot_be_claimed_for_a_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp)
            registry.add_traveler(_identity("Wren"), position=(0, 0))
            registry.remove_traveler("Wren")
            self.assertFalse(registry.try_occupy_pair("Mara", "Wren"))

    def test_move_tool_after_leaving_is_harmless(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp)
            agent = registry.add_traveler(_identity("Wren"), position=(0, 0))
            registry.remove_traveler("Wren")
            self.assertEqual(agent.tools.execute("move_to", {"x": 5, "y": 5}), "You've already left town.")

    def test_travelers_are_never_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, inventory_dir=Path(tmp) / "inventory")
            agent = registry.add_traveler(_identity("Wren"), position=(0, 0))
            agent.memory.add("met Mara", importance=5)
            registry.save_all()

            self.assertTrue((Path(tmp) / "memory" / "Mara.json").exists())
            self.assertFalse((Path(tmp) / "memory" / "Wren.json").exists())
            self.assertFalse((Path(tmp) / "inventory" / "Wren.json").exists())

    def test_holding_brakes_a_walking_traveler(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp)
            agent = registry.add_traveler(_identity("Wren"), position=(-100, 0))
            agent.destination = (100, 0)
            for _ in range(30):
                registry.step_movement(1 / 30)
            self.assertGreater(agent.velocity[0], 0)

            for _ in range(30):
                registry.step_movement(1 / 30, holding={"Wren"})
            self.assertEqual(agent.velocity, (0.0, 0.0))
            self.assertEqual(agent.destination, (100, 0))  # kept, not forgotten

    def test_resident_can_buy_from_a_traveler(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp)
            registry.get("Mara").inventory.money = 10
            registry.add_traveler(_identity("Wren", starting_items={"map": 1}), position=(0, 0))

            result = registry.get("Mara").tools.execute(
                "buy_item", {"item": "map", "seller_name": "Wren", "total_price": 4}
            )
            self.assertIn("You bought", result)
            self.assertEqual(registry.get("Wren").inventory.money, 4)


class TravelerConversationTests(unittest.TestCase):
    def test_traveler_can_start_a_conversation_and_its_lines_stay_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            long_line = " ".join(f"word{i}" for i in range(30))
            llm = _RecordingLLM(ToolCall(name=SPEAK_TOOL_NAME, arguments={"text": long_line}))
            registry = _registry(tmp, llm, residents=(("Mara", (0, 0)),))
            wren = registry.add_traveler(_identity("Wren"), position=(3, 0))

            result = wren.tools.execute("initiate_conversation", {"target_name": "Mara"})

            self.assertTrue(result.startswith("You had a conversation with Mara"))
            wren_lines = [m.content for m in wren.memory.all()]
            self.assertTrue(wren_lines)
            for line in wren_lines:
                said = line.split(" -> ", 1)[1]
                self.assertLessEqual(len(said.split()), NPCRegistry.TRAVELER_MAX_WORDS)
            self.assertFalse(registry.is_busy("Wren"))

    def test_traveler_out_of_range_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp)
            wren = registry.add_traveler(_identity("Wren"), position=(-100, 0))
            result = wren.tools.execute("initiate_conversation", {"target_name": "Mara"})
            self.assertIn("units away", result)


class TravelerPromptTests(unittest.TestCase):
    def test_prompt_has_traveler_instructions_profile_and_exit_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _RecordingLLM()
            registry = _registry(tmp, llm, instructions_path=INSTRUCTIONS_PATH)
            agent = registry.add_traveler(
                _identity("Wren"), position=(0, 0), standing_context="Your exit point: (100, 7)."
            )
            agent.respond("hello")

            self.assertIn("a traveler passing through town", llm.last_system)
            self.assertIn("Your exit point: (100, 7).", llm.last_system)
            self.assertIn(f"at most {NPCRegistry.TRAVELER_MAX_WORDS} words", llm.last_system)
            self.assertNotIn("{max_words}", llm.last_system)
            self.assertNotIn("Workplace:", llm.last_system)

    def test_residents_keep_their_own_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _RecordingLLM()
            registry = _registry(tmp, llm, instructions_path=INSTRUCTIONS_PATH)
            registry.get("Mara").respond("hello")
            self.assertIn("Workplace:", llm.last_system)
            self.assertNotIn("passing through", llm.last_system)

    def test_traveler_speech_is_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            long_line = " ".join(f"word{i}" for i in range(30))
            llm = _RecordingLLM(ToolCall(name=SPEAK_TOOL_NAME, arguments={"text": long_line}))
            registry = _registry(tmp, llm)
            agent = registry.add_traveler(_identity("Wren"), position=(0, 0))

            utterance = agent.respond("hello").utterance
            self.assertEqual(len(utterance.split()), NPCRegistry.TRAVELER_MAX_WORDS)
            self.assertTrue(utterance.endswith("..."))
            speak = next(t for t in llm.last_tools if t["name"] == SPEAK_TOOL_NAME)
            self.assertIn(f"at most {NPCRegistry.TRAVELER_MAX_WORDS} words", speak["description"])

    def test_resident_speech_is_not_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            long_line = " ".join(f"word{i}" for i in range(30))
            llm = _RecordingLLM(ToolCall(name=SPEAK_TOOL_NAME, arguments={"text": long_line}))
            registry = _registry(tmp, llm)
            self.assertEqual(registry.get("Mara").respond("hello").utterance, long_line)


if __name__ == "__main__":
    unittest.main()
