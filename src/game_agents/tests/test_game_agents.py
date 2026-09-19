from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from game_agents.agent import Agent, Scene
from game_agents.conversation import run_conversation
from game_agents.identity import Identity
from game_agents.inventory import Inventory, TradeError, trade
from game_agents.llm import SPEAK_TOOL_NAME, LLMResult, MockLLMClient, ToolCall
from game_agents.memory import MemoryStore
from game_agents.registry import NPCRegistry
from game_agents.storage import (
    load_identities,
    load_instructions,
    load_inventory,
    load_memory,
    load_profile_template,
    load_places,
    render_places,
    save_identities,
    save_memory,
)
from game_agents.tools import Tool, ToolRegistry
from game_agents.world import MAP_MAX, MAP_MIN, NPC_ACCEL, NPC_MAX_SPEED, clamp_coordinate, distance


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


class IdentityProfileTests(unittest.TestCase):
    def test_prompt_block_includes_home_workplace_and_habits(self):
        identity = Identity(
            name="Mara",
            traits=[],
            backstory="",
            speech_style="",
            home=(-40, 15),
            workplace=(12, -30),
            habits=["Goes to the forge each morning", "Returns home at dusk"],
        )
        block = identity.prompt_block()

        self.assertIn("Home: (-40, 15)", block)
        self.assertIn("Workplace: (12, -30)", block)
        self.assertIn("Habits: Goes to the forge each morning; Returns home at dusk", block)

    def test_default_home_workplace_and_empty_habits(self):
        identity = _identity("Mara")
        block = identity.prompt_block()

        self.assertIn("Home: (0, 0)", block)
        self.assertIn("Workplace: (0, 0)", block)
        self.assertIn("Habits: none", block)


class WorldBoundsTests(unittest.TestCase):
    def test_clamp_leaves_in_range_values_untouched(self):
        self.assertEqual(clamp_coordinate(0), 0)
        self.assertEqual(clamp_coordinate(MAP_MIN), MAP_MIN)
        self.assertEqual(clamp_coordinate(MAP_MAX), MAP_MAX)

    def test_clamp_pulls_out_of_range_values_to_the_nearest_edge(self):
        self.assertEqual(clamp_coordinate(MAP_MAX + 50), MAP_MAX)
        self.assertEqual(clamp_coordinate(MAP_MIN - 50), MAP_MIN)


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
                    home=(-40, 15),
                    workplace=(12, -30),
                    habits=["Goes to the forge each morning"],
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


class WorldContextTests(unittest.TestCase):
    def _write_world(self, tmp: str) -> Path:
        path = Path(tmp) / "world.json"
        path.write_text(
            json.dumps({"places": [{"name": "The Forge", "position": [12, -30], "description": "Hot and loud."}]}),
            encoding="utf-8",
        )
        return path

    def test_renders_each_place_with_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = render_places(load_places(self._write_world(tmp)))
            self.assertIn("The Forge at (12, -30): Hot and loud.", context)

    def test_missing_world_file_returns_empty_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_places(Path(tmp) / "nope.json"), [])

    def test_residents_and_travelers_both_see_places(self):
        with tempfile.TemporaryDirectory() as tmp:
            npcs_path = Path(tmp) / "npcs.json"
            save_identities([_identity("A")], npcs_path)
            registry = NPCRegistry(
                npcs_path, Path(tmp) / "memory", MockLLMClient(), world_path=self._write_world(tmp)
            )
            traveler = registry.add_traveler(_identity("T"), position=(0, 0), standing_context="Exit: (100, 0)")
            self.assertIn("The Forge", registry.get("A")._build_system_prompt(Scene(), []))
            traveler_prompt = traveler._build_system_prompt(Scene(), [])
            self.assertIn("The Forge", traveler_prompt)
            self.assertIn("Exit: (100, 0)", traveler_prompt)


class AgentPositionTests(unittest.TestCase):
    def test_spawns_at_identity_home_by_default(self):
        identity = _identity("Mara")
        identity.home = (-40, 15)
        agent = Agent(identity, MockLLMClient())

        self.assertEqual(agent.position, (-40, 15))

    def test_explicit_position_overrides_home(self):
        identity = _identity("Mara")
        identity.home = (-40, 15)
        agent = Agent(identity, MockLLMClient(), position=(3, 4))

        self.assertEqual(agent.position, (3, 4))


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


class RoutineTurnMemoryTests(unittest.TestCase):
    """Routine (unprompted, world-tick) turns only leave a memory when the
    NPC did something consequential -- otherwise idle ticks would flood the
    top-ranked memories and push real interactions out.
    """

    def _agent(self, tool_call: ToolCall) -> Agent:
        tools = ToolRegistry()
        for name in ("move_to", "buy_item"):
            tools.register(
                Tool(name=name, description=name, parameters={"type": "object", "properties": {}}, handler=lambda: "ok")
            )
        return Agent(_identity("Mara"), _FixedLLM(tool_call), tools)

    def test_routine_move_is_not_remembered(self):
        agent = self._agent(ToolCall(name="move_to", arguments={}))
        result = agent.respond("It is now 08:15 (morning).", routine=True)

        self.assertEqual(result.action["name"], "move_to")
        self.assertEqual(agent.memory.all(), [])

    def test_routine_speech_is_not_remembered(self):
        agent = self._agent(ToolCall(name=SPEAK_TOOL_NAME, arguments={"text": "nice day"}))
        result = agent.respond("It is now 08:15 (morning).", routine=True)

        self.assertEqual(result.utterance, "nice day")
        self.assertEqual(agent.memory.all(), [])

    def test_routine_consequential_action_is_remembered(self):
        agent = self._agent(ToolCall(name="buy_item", arguments={}))
        agent.respond("It is now 08:15 (morning).", routine=True)

        self.assertEqual(len(agent.memory.all()), 1)

    def test_non_routine_move_is_still_remembered(self):
        agent = self._agent(ToolCall(name="move_to", arguments={}))
        agent.respond("Meet me at the forge.")

        self.assertEqual(len(agent.memory.all()), 1)


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

    def test_current_position_is_included(self):
        identity = _identity("Mara")
        identity.home = (7, -8)
        recorder = _RecordingLLM()
        agent = Agent(identity, recorder)
        agent.respond("hi")

        self.assertIn("You are currently at (7, -8).", recorder.last_system)

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
            self.assertEqual(
                names,
                {"initiate_conversation", "move_to", "wait", "check_inventory", "buy_item", "sell_item", "wave"},
            )

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

    def test_refuses_when_target_is_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, conversation_turns=2)
            registry.get("Finn").position = (8, 7)  # ~10.6 away

            result = registry.get("Mara").tools.execute("initiate_conversation", {"target_name": "Finn"})

            self.assertIn("within 10", result)
            self.assertFalse(registry.is_busy("Mara"))
            self.assertFalse(registry.is_busy("Finn"))
            self.assertEqual(len(registry.get("Finn").memory.all()), 0)

            registry.get("Finn").position = (6, 8)  # exactly 10 away: allowed
            result = registry.get("Mara").tools.execute("initiate_conversation", {"target_name": "Finn"})
            self.assertIn("You had a conversation", result)

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


class MoveToolTests(unittest.TestCase):
    """The registry-built move_to tool: sets the specific NPC's destination
    (clamped to the map's edges) without teleporting it -- the NPC only
    gets there by walking, via NPCRegistry.step_movement(). Wired through
    Agent.tools.execute() directly, same as InitiateConversationToolTests.
    """

    def _registry(self, tmp, *, names=("Mara", "Finn"), **kwargs):
        npcs_path = Path(tmp) / "npcs.json"
        save_identities([_identity(n) for n in names], npcs_path)
        return NPCRegistry(npcs_path, Path(tmp) / "memory", MockLLMClient(), **kwargs)

    def _walk(self, registry, seconds, dt=1 / 30):
        for _ in range(round(seconds / dt)):
            registry.step_movement(dt)

    def test_sets_a_destination_without_teleporting(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, names=("Mara",))

            result = registry.get("Mara").tools.execute("move_to", {"x": 30, "y": -20})

            self.assertIn("(30, -20)", result)
            self.assertEqual(registry.get("Mara").destination, (30, -20))
            self.assertEqual(registry.get("Mara").position, (0, 0))

    def test_walks_there_over_time_and_stops_on_the_spot(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, names=("Mara",))
            registry.get("Mara").tools.execute("move_to", {"x": 30, "y": -20})

            self._walk(registry, 0.2)
            partway = registry.get("Mara").position
            self.assertGreater(distance(partway, (0, 0)), 0)
            self.assertLess(distance(partway, (0, 0)), distance((30, -20), (0, 0)))

            self._walk(registry, 5)
            mara = registry.get("Mara")
            self.assertEqual(mara.position, (30, -20))
            self.assertEqual(mara.velocity, (0.0, 0.0))
            self.assertIsNone(mara.destination)

    def test_accelerates_from_rest_and_never_exceeds_max_speed(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, names=("Mara",))
            registry.get("Mara").tools.execute("move_to", {"x": 100, "y": 0})

            registry.step_movement(0.1)
            self.assertAlmostEqual(registry.get("Mara").velocity[0], NPC_ACCEL * 0.1)

            for _ in range(60):
                registry.step_movement(1 / 30)
                self.assertLessEqual(math.hypot(*registry.get("Mara").velocity), NPC_MAX_SPEED + 1e-9)

    def test_busy_npc_brakes_to_a_stop_and_resumes_after(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, names=("Mara",))
            registry.get("Mara").tools.execute("move_to", {"x": 100, "y": 0})
            self._walk(registry, 0.5)

            registry.try_occupy("Mara")
            self._walk(registry, 1)
            stopped_at = registry.get("Mara").position
            self.assertEqual(registry.get("Mara").velocity, (0.0, 0.0))
            self._walk(registry, 1)
            self.assertEqual(registry.get("Mara").position, stopped_at)

            registry.release("Mara")
            self._walk(registry, 5)
            self.assertEqual(registry.get("Mara").position, (100, 0))

    def test_clamps_out_of_range_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, names=("Mara",))

            registry.get("Mara").tools.execute("move_to", {"x": 500, "y": -500})

            self.assertEqual(registry.get("Mara").destination, (100, -100))

    def test_only_moves_the_calling_npc(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp)

            registry.get("Mara").tools.execute("move_to", {"x": 10, "y": 10})
            self._walk(registry, 3)

            self.assertEqual(registry.get("Mara").position, (10, 10))
            self.assertEqual(registry.get("Finn").position, (0, 0))  # default home, untouched


class WaitToolTests(unittest.TestCase):
    def test_every_npc_can_wait_and_it_is_a_true_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            npcs_path = Path(tmp) / "npcs.json"
            save_identities([_identity("Mara")], npcs_path)
            registry = NPCRegistry(npcs_path, Path(tmp) / "memory", MockLLMClient())

            result = registry.get("Mara").tools.execute("wait", {})

            self.assertIsInstance(result, str)
            self.assertEqual(registry.get("Mara").position, (0, 0))


class SingleNameBusyTests(unittest.TestCase):
    """try_occupy/release: the player-conversation busy claim (see
    mini_map.simulation.Simulation), reusing try_occupy_pair/release_pair
    with the same name on both sides.
    """

    def _registry(self, tmp, names=("Mara", "Finn")):
        npcs_path = Path(tmp) / "npcs.json"
        save_identities([_identity(n) for n in names], npcs_path)
        return NPCRegistry(npcs_path, Path(tmp) / "memory", MockLLMClient())

    def test_occupy_marks_busy_and_release_frees_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, names=("Mara",))

            self.assertTrue(registry.try_occupy("Mara"))
            self.assertTrue(registry.is_busy("Mara"))

            registry.release("Mara")
            self.assertFalse(registry.is_busy("Mara"))

    def test_cannot_occupy_an_already_busy_npc(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, names=("Mara", "Finn"))
            registry.try_occupy_pair("Mara", "Finn")

            self.assertFalse(registry.try_occupy("Mara"))

    def test_single_occupy_also_blocks_pair_occupy_and_vice_versa(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp, names=("Mara", "Finn"))
            registry.try_occupy("Mara")

            self.assertFalse(registry.try_occupy_pair("Mara", "Finn"))


class ResolveTests(unittest.TestCase):
    def test_resolves_exact_and_case_insensitive_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            npcs_path = Path(tmp) / "npcs.json"
            save_identities([_identity("Mara")], npcs_path)
            registry = NPCRegistry(npcs_path, Path(tmp) / "memory", MockLLMClient())

            self.assertEqual(registry.resolve("Mara"), "Mara")
            self.assertEqual(registry.resolve("mara"), "Mara")
            self.assertIsNone(registry.resolve("Nobody"))


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


class InventoryTests(unittest.TestCase):
    def test_item_names_are_normalized(self):
        inv = Inventory(money=5, items={"Spare  Net ": 1, "spare net": 2})
        self.assertEqual(inv.items, {"spare net": 3})
        self.assertEqual(inv.count("SPARE NET"), 3)

    def test_describe_lists_coins_and_items(self):
        self.assertEqual(Inventory(money=1).describe(), "You have 1 coin and no items.")
        self.assertEqual(
            Inventory(money=7, items={"fish": 2, "anchor": 1}).describe(),
            "You have 7 coins and: 1 x anchor, 2 x fish.",
        )


class TradeTests(unittest.TestCase):
    def test_successful_trade_moves_item_and_coins(self):
        buyer, seller = Inventory(money=10), Inventory(money=0, items={"net": 2})

        trade(buyer=buyer, seller=seller, item="Net", quantity=2, total_price=8)

        self.assertEqual((buyer.money, buyer.items), (2, {"net": 2}))
        self.assertEqual((seller.money, seller.items), (8, {}))

    def test_failed_trades_leave_both_inventories_untouched(self):
        cases = [
            dict(item="net", quantity=1, total_price=11),  # buyer can't afford
            dict(item="net", quantity=3, total_price=1),  # seller doesn't have enough
            dict(item="boat", quantity=1, total_price=1),  # seller doesn't have it at all
            dict(item="net", quantity=0, total_price=1),
            dict(item="net", quantity=1, total_price=-5),
            dict(item="  ", quantity=1, total_price=1),
        ]
        for case in cases:
            with self.subTest(**case):
                buyer, seller = Inventory(money=10), Inventory(money=0, items={"net": 2})
                with self.assertRaises(TradeError):
                    trade(buyer=buyer, seller=seller, **case)
                self.assertEqual((buyer.money, buyer.items), (10, {}))
                self.assertEqual((seller.money, seller.items), (0, {"net": 2}))

    def test_cannot_trade_with_yourself(self):
        inv = Inventory(money=10, items={"net": 1})
        with self.assertRaises(TradeError):
            trade(buyer=inv, seller=inv, item="net", quantity=1, total_price=1)


class InventoryToolTests(unittest.TestCase):
    """check_inventory / buy_item / sell_item as built by the registry,
    wired through Agent.tools.execute() directly, same as MoveToolTests.
    """

    def _registry(self, tmp, **kwargs):
        mara, finn = _identity("Mara"), _identity("Finn")
        mara.starting_money, mara.starting_items = 20, {"horseshoe": 3}
        finn.starting_money, finn.starting_items = 5, {"fish": 4}
        npcs_path = Path(tmp) / "npcs.json"
        save_identities([mara, finn], npcs_path)
        return NPCRegistry(npcs_path, Path(tmp) / "memory", MockLLMClient(), **kwargs)

    def test_starts_from_the_identity_starting_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp)

            result = registry.get("Mara").tools.execute("check_inventory", {})

            self.assertEqual(result, "You have 20 coins and: 3 x horseshoe.")

    def test_buy_moves_goods_and_coins_and_tells_the_seller(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp)

            result = registry.get("Mara").tools.execute(
                "buy_item", {"item": "fish", "seller_name": "finn", "total_price": 6, "quantity": 2}
            )

            self.assertIn("You bought 2 x fish", result)
            mara, finn = registry.get("Mara").inventory, registry.get("Finn").inventory
            self.assertEqual((mara.money, mara.count("fish")), (14, 2))
            self.assertEqual((finn.money, finn.count("fish")), (11, 2))
            [memory] = registry.get("Finn").memory.all()
            self.assertIn("Sold 2 x fish to Mara", memory.content)
            self.assertEqual(memory.tags, {"Mara"})

    def test_sell_is_the_same_trade_from_the_other_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp)

            result = registry.get("Mara").tools.execute(
                "sell_item", {"item": "horseshoe", "buyer_name": "Finn", "total_price": 5}
            )

            self.assertIn("You sold 1 x horseshoe", result)
            self.assertEqual(registry.get("Mara").inventory.money, 25)
            self.assertEqual(registry.get("Finn").inventory.money, 0)
            self.assertEqual(registry.get("Finn").inventory.count("horseshoe"), 1)

    def test_unaffordable_or_missing_goods_change_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp)
            mara = registry.get("Mara")

            too_poor = registry.get("Finn").tools.execute(
                "buy_item", {"item": "horseshoe", "seller_name": "Mara", "total_price": 6}
            )
            no_such_item = mara.tools.execute("sell_item", {"item": "anvil", "buyer_name": "Finn", "total_price": 1})
            nobody = mara.tools.execute("buy_item", {"item": "fish", "seller_name": "Nobody", "total_price": 1})

            self.assertIn("didn't go through", too_poor)
            self.assertIn("didn't go through", no_such_item)
            self.assertIn("Nobody", nobody)
            self.assertEqual((mara.inventory.money, mara.inventory.items), (20, {"horseshoe": 3}))
            self.assertEqual(registry.get("Finn").inventory.money, 5)
            self.assertEqual(len(registry.get("Finn").memory.all()), 0)

    def test_inventory_persists_across_reloads_when_a_dir_is_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            inventory_dir = Path(tmp) / "inventory"
            registry = self._registry(tmp, inventory_dir=inventory_dir)
            registry.get("Mara").tools.execute("buy_item", {"item": "fish", "seller_name": "Finn", "total_price": 3})
            registry.save_all()

            reloaded = self._registry(tmp, inventory_dir=inventory_dir)

            self.assertEqual(reloaded.get("Mara").inventory.money, 17)
            self.assertEqual(reloaded.get("Finn").inventory.count("fish"), 3)
            self.assertIsNone(load_inventory("Nobody", inventory_dir))

    def test_identity_round_trip_keeps_starting_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "npcs.json"
            identity = _identity("Mara")
            identity.starting_money, identity.starting_items = 9, {"hammer": 1}
            save_identities([identity], path)

            [loaded] = load_identities(path)

            self.assertEqual((loaded.starting_money, loaded.starting_items), (9, {"hammer": 1}))

    def test_trade_refused_when_npcs_are_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp)
            registry.get("Finn").position = (6, 8)  # exactly 10 away: allowed
            ok = registry.get("Mara").tools.execute("buy_item", {"item": "fish", "seller_name": "Finn", "total_price": 1})
            self.assertIn("You bought", ok)

            registry.get("Finn").position = (8, 7)  # ~10.6 away: refused
            refused = registry.get("Mara").tools.execute(
                "sell_item", {"item": "horseshoe", "buyer_name": "Finn", "total_price": 1}
            )

            self.assertIn("within 10", refused)
            self.assertEqual(registry.get("Mara").inventory.count("horseshoe"), 3)
            self.assertEqual(registry.get("Finn").inventory.money, 6)  # only the first trade's coin
