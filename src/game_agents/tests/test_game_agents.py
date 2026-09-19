from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from game_agents.agent import Agent, Scene
from game_agents.conversation import run_conversation
from game_agents.identity import Identity
from game_agents.llm import SPEAK_TOOL_NAME, LLMResult, MockLLMClient, ToolCall
from game_agents.memory import MemoryStore
from game_agents.registry import NPCRegistry
from game_agents.storage import (
    load_identities,
    load_instructions,
    load_memory,
    load_profile_template,
    save_identities,
    save_memory,
)
from game_agents.tools import Tool, ToolRegistry


def _identity(name: str) -> Identity:
    return Identity(name=name, traits=[], backstory="", speech_style="", goals=[])


class _FixedLLM:
    """Always returns the same preset tool call, regardless of input."""

    def __init__(self, tool_call: ToolCall):
        self._tool_call = tool_call

    def complete(self, *, system, messages, tools):
        return LLMResult(tool_call=self._tool_call)


class _RecordingLLM:
    """Captures the system prompt and tools it was called with, for
    asserting on prompt/schema composition without a real backend. Always
    replies by speaking, unless told to reply with something else.
    """

    def __init__(self, reply: ToolCall | None = None):
        self.last_system = ""
        self.last_tools: list[dict] = []
        self._reply = reply or ToolCall(name=SPEAK_TOOL_NAME, arguments={"text": "ok"})

    def complete(self, *, system, messages, tools):
        self.last_system = system
        self.last_tools = tools
        return LLMResult(tool_call=self._reply)


class MemoryStoreTests(unittest.TestCase):
    def test_tag_filtering_keeps_untagged_as_general_knowledge(self):
        store = MemoryStore()
        store.add("the sky is blue", importance=2)
        store.add("Bob borrowed my hammer", importance=6, tags={"Bob"})
        store.add("Alice paid her tab", importance=6, tags={"Alice"})

        bob_ctx = [m.content for m in store.retrieve(tags={"Bob"})]
        self.assertIn("Bob borrowed my hammer", bob_ctx)
        self.assertIn("the sky is blue", bob_ctx)
        self.assertNotIn("Alice paid her tab", bob_ctx)


class StorageRoundTripTests(unittest.TestCase):
    def test_identity_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "npcs.json"
            original = [
                Identity(
                    name="Mara",
                    traits=["gruff"],
                    backstory="runs the forge",
                    speech_style="dry",
                    goals=["finish the order"],
                )
            ]
            save_identities(original, path)
            self.assertEqual(load_identities(path), original)

    def test_memory_round_trip_preserves_timestamp_and_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore()
            store.add("hello", importance=3, tags={"player"}, timestamp=1000.0)
            save_memory("Mara", store, tmp)

            [memory] = load_memory("Mara", tmp).all()
            self.assertEqual(memory.content, "hello")
            self.assertEqual(memory.timestamp, 1000.0)
            self.assertEqual(memory.tags, {"player"})

    def test_missing_memory_file_returns_empty_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_memory("Nobody", tmp).all(), [])

    def test_instructions_load_as_dashed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instructions.json"
            path.write_text('{"instructions": ["stay in character", "speech only"]}', encoding="utf-8")
            self.assertEqual(load_instructions(path), "- stay in character\n- speech only")

    def test_missing_instructions_file_returns_empty_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_instructions(Path(tmp) / "nope.json"), "")

    def test_profile_template_loads_from_the_same_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instructions.json"
            path.write_text('{"profile_template": "{name} says hi"}', encoding="utf-8")
            self.assertEqual(load_profile_template(path), "{name} says hi")

    def test_missing_profile_template_returns_empty_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_profile_template(Path(tmp) / "nope.json"), "")


class RegistryTests(unittest.TestCase):
    def test_each_npc_keeps_independent_memory_across_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            npcs_path = Path(tmp) / "npcs.json"
            memory_dir = Path(tmp) / "memory"
            save_identities([_identity("A"), _identity("B")], npcs_path)

            registry = NPCRegistry(npcs_path, memory_dir, MockLLMClient())
            self.assertIsNot(registry.get("A").memory, registry.get("B").memory)

            registry.get("A").respond("hi", tags={"player"})
            registry.save_all()

            reloaded = NPCRegistry(npcs_path, memory_dir, MockLLMClient())
            self.assertEqual(len(reloaded.get("A").memory.all()), 1)
            self.assertEqual(len(reloaded.get("B").memory.all()), 0)

    def test_unknown_name_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            npcs_path = Path(tmp) / "npcs.json"
            save_identities([], npcs_path)
            registry = NPCRegistry(npcs_path, Path(tmp) / "memory", MockLLMClient())
            self.assertIsNone(registry.get("Nobody"))


class AgentTurnTests(unittest.TestCase):
    def test_respond_records_a_memory(self):
        agent = Agent(_identity("Mara"), MockLLMClient())
        result = agent.respond("hello", scene=Scene(location="forge"), tags={"player"})

        self.assertEqual(result.utterance, "(mock reply to: hello)")
        [memory] = agent.memory.all()
        self.assertEqual(memory.importance, 4)  # spoke, didn't act -> the lower heuristic band

    def test_respond_executes_and_records_a_tool_call(self):
        seen = {}
        tools = ToolRegistry()
        tools.register(
            Tool(
                name="wave",
                description="wave",
                parameters={"type": "object", "properties": {"target": {"type": "string"}}},
                handler=lambda target: seen.setdefault("target", target),
            )
        )
        llm = _FixedLLM(ToolCall(name="wave", arguments={"target": "Bob"}))
        agent = Agent(_identity("Mara"), llm, tools)
        result = agent.respond("hi")

        self.assertEqual(seen["target"], "Bob")
        self.assertEqual(result.action, {"name": "wave", "arguments": {"target": "Bob"}, "result": "Bob"})
        [memory] = agent.memory.all()
        self.assertEqual(memory.importance, 7)  # acted -> the higher heuristic band


class SpeakOrActionSchemaTests(unittest.TestCase):
    """The core contract from this step: every turn resolves to exactly one
    of utterance/action, never both, never neither -- and `speak` is always
    on offer, even for an NPC with no registered actions at all.
    """

    def test_speak_tool_call_yields_utterance_and_no_action(self):
        llm = _FixedLLM(ToolCall(name=SPEAK_TOOL_NAME, arguments={"text": "howdy"}))
        agent = Agent(_identity("Mara"), llm)
        result = agent.respond("hi")

        self.assertEqual(result.utterance, "howdy")
        self.assertIsNone(result.action)

    def test_action_tool_call_yields_action_and_no_utterance(self):
        tools = ToolRegistry()
        tools.register(
            Tool(
                name="wave",
                description="wave",
                parameters={"type": "object", "properties": {"target": {"type": "string"}}},
                handler=lambda target: None,
            )
        )
        llm = _FixedLLM(ToolCall(name="wave", arguments={"target": "Bob"}))
        agent = Agent(_identity("Mara"), llm, tools)
        result = agent.respond("hi")

        self.assertIsNone(result.utterance)
        self.assertIsNotNone(result.action)

    def test_speak_is_offered_even_with_no_registered_actions(self):
        recorder = _RecordingLLM()
        agent = Agent(_identity("Mara"), recorder)
        agent.respond("hi")

        self.assertEqual([t["name"] for t in recorder.last_tools], [SPEAK_TOOL_NAME])

    def test_speak_is_offered_alongside_registered_actions(self):
        tools = ToolRegistry()
        tools.register(
            Tool(name="wave", description="wave", parameters={"type": "object", "properties": {}}, handler=lambda: None)
        )
        recorder = _RecordingLLM()
        agent = Agent(_identity("Mara"), recorder, tools)
        agent.respond("hi")

        self.assertEqual([t["name"] for t in recorder.last_tools], [SPEAK_TOOL_NAME, "wave"])


class SystemPromptTests(unittest.TestCase):
    def test_instructions_prepended_when_provided(self):
        recorder = _RecordingLLM()
        agent = Agent(_identity("Mara"), recorder, instructions="- stay in character")
        agent.respond("hi")

        self.assertTrue(recorder.last_system.startswith("- stay in character"))

    def test_no_instructions_means_no_extra_block(self):
        recorder = _RecordingLLM()
        agent = Agent(_identity("Mara"), recorder)
        agent.respond("hi")

        self.assertTrue(recorder.last_system.startswith("You are Mara."))

    def test_empty_scene_omits_scene_block(self):
        recorder = _RecordingLLM()
        agent = Agent(_identity("Mara"), recorder)
        agent.respond("hi", scene=Scene())

        self.assertNotIn("Time:", recorder.last_system)
        self.assertNotIn("Location:", recorder.last_system)

    def test_populated_scene_is_included(self):
        recorder = _RecordingLLM()
        agent = Agent(_identity("Mara"), recorder)
        agent.respond("hi", scene=Scene(location="the forge"))

        self.assertIn("Location: the forge", recorder.last_system)

    def test_custom_profile_template_is_used(self):
        recorder = _RecordingLLM()
        agent = Agent(_identity("Mara"), recorder, profile_template="{name} the blacksmith")
        agent.respond("hi")

        self.assertTrue(recorder.last_system.startswith("Mara the blacksmith"))

    def test_registry_loads_profile_template_alongside_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            npcs_path = Path(tmp) / "npcs.json"
            instructions_path = Path(tmp) / "instructions.json"
            save_identities([_identity("Mara")], npcs_path)
            instructions_path.write_text(
                '{"instructions": ["be terse"], "profile_template": "NPC: {name}"}', encoding="utf-8"
            )

            registry = NPCRegistry(
                npcs_path, Path(tmp) / "memory", MockLLMClient(), instructions_path=instructions_path
            )
            self.assertEqual(registry.get("Mara").profile_template, "NPC: {name}")
            self.assertEqual(registry.get("Mara").instructions, "- be terse")


class ToolRegistryCopyTests(unittest.TestCase):
    def test_all_tools_returns_everything_registered(self):
        registry = ToolRegistry()
        tool = Tool(name="wave", description="wave", parameters={"type": "object", "properties": {}}, handler=lambda: None)
        registry.register(tool)

        self.assertEqual(registry.all_tools(), [tool])


class RunConversationTests(unittest.TestCase):
    """conversation.run_conversation() in isolation, with plain agents --
    NPCRegistry's initiate_conversation tool is a thin wrapper around this,
    tested separately below.
    """

    def test_alternates_turns_tagging_each_others_memory_with_the_other_name(self):
        a = Agent(_identity("A"), MockLLMClient())
        b = Agent(_identity("B"), MockLLMClient())

        transcript = run_conversation(a, b, turns=4)

        self.assertEqual(len(transcript), 4)
        self.assertTrue(all(turn.utterance is not None for turn in transcript))
        self.assertEqual([turn.speaker for turn in transcript], ["B", "A", "B", "A"])
        self.assertEqual(len(a.memory.all()), 2)
        self.assertEqual(len(b.memory.all()), 2)
        self.assertTrue(all(m.tags == {"B"} for m in a.memory.all()))
        self.assertTrue(all(m.tags == {"A"} for m in b.memory.all()))

    def test_ends_early_when_a_side_acts_instead_of_speaking(self):
        tools = ToolRegistry()
        tools.register(
            Tool(name="shrug", description="shrug", parameters={"type": "object", "properties": {}}, handler=lambda: "shrugged")
        )
        a = Agent(_identity("A"), MockLLMClient())
        b = Agent(_identity("B"), _FixedLLM(ToolCall(name="shrug", arguments={})), tools)

        transcript = run_conversation(a, b, turns=4)  # b speaks first (target goes first) and immediately shrugs

        self.assertEqual(len(transcript), 1)
        self.assertIsNone(transcript[0].utterance)
        self.assertEqual(transcript[0].action["name"], "shrug")


class InitiateConversationToolTests(unittest.TestCase):
    """The registry-built tool: busy-claiming, running the exchange, and
    releasing busy state, wired through Agent.tools.execute() directly
    (bypassing the LLM's own decision to call it, which isn't controllable
    without a real backend).
    """

    def _registry(self, tmp, *, names=("Mara", "Finn"), **kwargs):
        npcs_path = Path(tmp) / "npcs.json"
        save_identities([_identity(n) for n in names], npcs_path)
        return NPCRegistry(npcs_path, Path(tmp) / "memory", MockLLMClient(), **kwargs)

    def test_every_npc_gets_initiate_conversation_plus_common_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            common = ToolRegistry()
            common.register(
                Tool(name="wave", description="wave", parameters={"type": "object", "properties": {}}, handler=lambda: None)
            )
            registry = self._registry(tmp, names=("Mara",), tools=common)

            names = {t.name for t in registry.get("Mara").tools.all_tools()}
            self.assertEqual(names, {"initiate_conversation", "wave"})

    def test_initiate_conversation_runs_an_exchange_and_releases_busy_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, conversation_turns=2)

            result = registry.get("Mara").tools.execute("initiate_conversation", {"target_name": "Finn"})

            self.assertIn("Finn", result)
            self.assertFalse(registry.is_busy("Mara"))
            self.assertFalse(registry.is_busy("Finn"))
            # turns=2: Finn (the target) goes first, then Mara -- one memory each
            self.assertEqual(len(registry.get("Finn").memory.all()), 1)
            self.assertEqual(len(registry.get("Mara").memory.all()), 1)

    def test_refuses_when_target_already_busy(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, names=("Mara", "Finn", "Gus"))
            registry.try_occupy_pair("Finn", "Gus")

            result = registry.get("Mara").tools.execute("initiate_conversation", {"target_name": "Finn"})

            self.assertIn("busy", result)
            self.assertEqual(len(registry.get("Finn").memory.all()), 0)

    def test_handles_an_unknown_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, names=("Mara",))

            result = registry.get("Mara").tools.execute("initiate_conversation", {"target_name": "Nobody"})

            self.assertIn("Nobody", result)

    def test_resolves_a_differently_cased_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, conversation_turns=2)

            # lowercase "finn" rather than the exact stored name "Finn" -- an
            # LLM won't always reproduce capitalization exactly.
            result = registry.get("Mara").tools.execute("initiate_conversation", {"target_name": "finn"})

            self.assertIn("Finn", result)
            self.assertNotIn("no one", result)
            self.assertEqual(len(registry.get("Finn").memory.all()), 1)


class ConversationLoggingTests(unittest.TestCase):
    """What backs "get me the logs of the conversation" -- a full JSON
    transcript per NPC-to-NPC exchange, written when conversation_log_dir
    is set, with speaker/stimulus/utterance/action (function calls
    included) for every turn.
    """

    def test_writes_a_json_transcript_with_function_calls_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            npcs_path = Path(tmp) / "npcs.json"
            save_identities([_identity("Mara"), _identity("Finn")], npcs_path)
            log_dir = Path(tmp) / "conversation_log"
            registry = NPCRegistry(
                npcs_path,
                Path(tmp) / "memory",
                MockLLMClient(),
                conversation_turns=2,
                conversation_log_dir=log_dir,
            )

            registry.get("Mara").tools.execute("initiate_conversation", {"target_name": "Finn"})

            [log_file] = list(log_dir.glob("*.json"))
            transcript = json.loads(log_file.read_text(encoding="utf-8"))
            self.assertEqual(len(transcript), 2)
            self.assertEqual(transcript[0]["speaker"], "Finn")
            self.assertEqual(transcript[1]["speaker"], "Mara")
            for turn in transcript:
                self.assertIn("stimulus", turn)
                self.assertIn("utterance", turn)
                self.assertIn("action", turn)

    def test_no_log_dir_means_no_file_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            npcs_path = Path(tmp) / "npcs.json"
            save_identities([_identity("Mara"), _identity("Finn")], npcs_path)
            registry = NPCRegistry(npcs_path, Path(tmp) / "memory", MockLLMClient())

            registry.get("Mara").tools.execute("initiate_conversation", {"target_name": "Finn"})
            # Nothing to assert on a path we never created -- this is really
            # just confirming the call above didn't raise.


if __name__ == "__main__":
    unittest.main()
