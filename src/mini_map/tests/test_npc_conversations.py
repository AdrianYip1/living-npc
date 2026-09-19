"""NPC-to-NPC conversations as the mini-map sees them: run off the world
tick's thread, paced for reading, and published line by line in state()'s
speech feed.
"""
from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from environment_agent.agent import EnvironmentAgent

from game_agents.conversation import ConversationTurn
from game_agents.identity import Identity
from game_agents.llm import ENDS_CONVERSATION_FIELD, SPEAK_TOOL_NAME, LLMResult, ToolCall
from game_agents.registry import NPCRegistry
from game_agents.storage import save_identities
from game_agents.world import IN_REACH_LABEL

from mini_map.simulation import Simulation


def _identity(name: str) -> Identity:
    return Identity(name=name, traits=[], backstory="", speech_style="", goals=[])


class _ChatLLM:
    """Outside a conversation, Mara starts one with Finn and everyone else
    waits. Inside one, each side says "<name> line <n>", and Finn says
    goodbye on his second line. `gate`, if given, has to be set before any
    conversation line is decided -- to hold an exchange open.
    """

    def __init__(self, gate: threading.Event | None = None):
        self._gate = gate
        self.systems: dict[str, list[str]] = {}
        self._lines: dict[str, int] = {}
        self._lock = threading.Lock()

    def complete(self, *, system, messages, tools):
        name = system.split(".", 1)[0].removeprefix("You are ")
        with self._lock:
            self.systems.setdefault(name, []).append(system)
        if ENDS_CONVERSATION_FIELD not in tools[0]["parameters"]["properties"]:
            if name == "Mara":
                return LLMResult(tool_call=ToolCall(name="initiate_conversation", arguments={"target_name": "Finn"}))
            return LLMResult(tool_call=ToolCall(name="wait", arguments={}))
        if self._gate is not None:
            self._gate.wait(timeout=5)
        with self._lock:
            n = self._lines[name] = self._lines.get(name, 0) + 1
        arguments = {"text": f"{name} line {n}"}
        if name == "Finn" and n == 2:
            arguments[ENDS_CONVERSATION_FIELD] = True
        return LLMResult(tool_call=ToolCall(name=SPEAK_TOOL_NAME, arguments=arguments))


class _MutterLLM:
    def complete(self, *, system, messages, tools):
        return LLMResult(tool_call=ToolCall(name=SPEAK_TOOL_NAME, arguments={"text": "Hm."}))


def _registry(tmp, llm, names=("Mara", "Finn", "Gus")) -> NPCRegistry:
    npcs_path = Path(tmp) / "npcs.json"
    save_identities([_identity(n) for n in names], npcs_path)
    registry = NPCRegistry(npcs_path, Path(tmp) / "memory", llm, conversation_turns=10)
    if "Gus" in names:
        registry.get("Gus").position = (60, 60)  # out of everyone's sight
    return registry


def _npc(sim: Simulation, name: str) -> dict:
    return next(npc for npc in sim.state()["npcs"] if npc["name"] == name)


class SpeechFeedTests(unittest.TestCase):
    def test_conversation_lines_land_in_the_feed_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = Simulation(_registry(tmp, _ChatLLM()), EnvironmentAgent(seed=1), line_pacing=False)

            sim.tick()
            sim.wait_for_pending()

            speech = sim.state()["speech"]
            self.assertEqual(
                [(s["speaker"], s["listener"], s["text"]) for s in speech],
                [
                    ("Mara", "Finn", "Mara line 1"),
                    ("Finn", "Mara", "Finn line 1"),
                    ("Mara", "Finn", "Mara line 2"),
                    ("Finn", "Mara", "Finn line 2"),
                    ("Mara", "Finn", "Mara line 3"),  # answering Finn's goodbye
                ],
            )
            self.assertEqual([s["id"] for s in speech], [1, 2, 3, 4, 5])
            self.assertTrue(speech[3]["ends_conversation"])
            self.assertTrue(all(s["read_seconds"] >= Simulation.MIN_LINE_SECONDS for s in speech))
            self.assertEqual(_npc(sim, "Finn")["activity"], "talked with Mara")

    def test_state_shows_who_is_talking_to_whom_while_it_lasts(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = threading.Event()
            registry = _registry(tmp, _ChatLLM(gate))
            sim = Simulation(registry, EnvironmentAgent(seed=1), line_pacing=False)

            sim.tick()  # returns with the exchange still waiting on the gate

            self.assertEqual(_npc(sim, "Mara")["talking_to"], "Finn")
            self.assertEqual(_npc(sim, "Finn")["talking_to"], "Mara")
            self.assertTrue(_npc(sim, "Finn")["busy"])
            self.assertIsNone(_npc(sim, "Gus")["talking_to"])

            gate.set()
            sim.wait_for_pending()
            self.assertIsNone(_npc(sim, "Mara")["talking_to"])
            self.assertFalse(_npc(sim, "Finn")["busy"])

    def test_no_one_speaks_outside_a_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            # _MutterLLM tries to speak anyway; it isn't offered, so it
            # doesn't happen.
            registry = _registry(tmp, _MutterLLM(), names=("Mara",))
            sim = Simulation(registry, EnvironmentAgent(seed=1))

            sim.tick()

            self.assertEqual(sim.state()["speech"], [])
            self.assertEqual(registry.get("Mara").memory.all(), [])

    def test_a_reply_to_the_player_is_published_to_the_player(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = Simulation(_registry(tmp, _MutterLLM(), names=("Mara",)), EnvironmentAgent(seed=1))
            sim.start_conversation("Mara")

            sim.say("Mara", "hello")

            [line] = sim.state()["speech"]
            self.assertEqual((line["speaker"], line["listener"]), ("Mara", "player"))


class ResidentAwarenessTests(unittest.TestCase):
    def test_resident_turn_lists_who_is_close_enough_to_talk_to(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _ChatLLM()
            registry = _registry(tmp, llm)
            registry.get("Mara").position = (0, 0)
            registry.get("Finn").position = (5, 0)
            sim = Simulation(registry, EnvironmentAgent(seed=1), line_pacing=False)

            sim.tick()
            sim.wait_for_pending()

            self.assertIn(f"{IN_REACH_LABEL}: Finn at (5, 0).", llm.systems["Mara"][0])
            self.assertNotIn("Gus", llm.systems["Mara"][0])


class ConversationPacingTests(unittest.TestCase):
    def _sim(self, tmp, llm) -> Simulation:
        sim = Simulation(_registry(tmp, llm), EnvironmentAgent(seed=1))
        sim.MIN_LINE_SECONDS = 0.2
        sim.SECONDS_PER_WORD = 0.0
        return sim

    def test_lines_are_spaced_out_for_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = self._sim(tmp, _ChatLLM())
            start = time.monotonic()

            sim.tick()
            sim.wait_for_pending()

            # Five lines, each held until the previous one has been up
            # 0.2s, then the pair lingers 0.2s after the last one.
            self.assertGreaterEqual(time.monotonic() - start, 5 * 0.2 - 0.05)

    def test_the_world_tick_does_not_wait_for_a_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = self._sim(tmp, _ChatLLM())
            start = time.monotonic()

            sim.tick()

            self.assertLess(time.monotonic() - start, 0.2)
            sim.wait_for_pending()

    def test_pausing_freezes_a_conversation_mid_exchange(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = self._sim(tmp, _ChatLLM())
            sim.pause()

            sim.tick()
            time.sleep(0.5)
            self.assertEqual(sim.state()["speech"], [])

            sim.resume()
            sim.wait_for_pending()
            self.assertEqual(len(sim.state()["speech"]), 5)


class SpeakDescriptionTests(unittest.TestCase):
    def test_outside_a_conversation_speaking_is_thinking_aloud(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, _MutterLLM(), names=("Mara",))
            mara = registry.get("Mara")
            self.assertIn("Think aloud", mara._speak_schema()["description"])
            self.assertNotIn("Think aloud", mara._speak_schema(conversation=True)["description"])

    def test_the_player_gets_an_answer_not_a_remark(self):
        class _Recording(_MutterLLM):
            tools = None

            def complete(self, *, system, messages, tools):
                _Recording.tools = tools
                return super().complete(system=system, messages=messages, tools=tools)

        with tempfile.TemporaryDirectory() as tmp:
            sim = Simulation(_registry(tmp, _Recording(), names=("Mara",)), EnvironmentAgent(seed=1))
            sim.start_conversation("Mara")
            sim.say("Mara", "hello")
            self.assertNotIn("Think aloud", _Recording.tools[0]["description"])


class _ToolsLLM(_ChatLLM):
    """_ChatLLM, also recording which tools each routine turn offered."""

    def __init__(self):
        super().__init__()
        self.routine_tools: dict[str, list[set[str]]] = {}

    def complete(self, *, system, messages, tools):
        names = {t["name"] for t in tools}
        if not any(ENDS_CONVERSATION_FIELD in t["parameters"].get("properties", {}) for t in tools):
            name = system.split(".", 1)[0].removeprefix("You are ")
            self.routine_tools.setdefault(name, []).append(names)
        return super().complete(system=system, messages=messages, tools=tools)


class _OvertakenLLM:
    """Gus gets pulled into a conversation while still deciding his turn --
    which comes out as `call` anyway."""

    def __init__(self, registry_ref: list, call: ToolCall):
        self._registry_ref, self._call = registry_ref, call

    def complete(self, *, system, messages, tools):
        if system.startswith("You are Gus"):
            self._registry_ref[0].try_occupy("Gus")
            return LLMResult(tool_call=self._call)
        return LLMResult(tool_call=ToolCall(name="wait", arguments={}))


class ConversationFollowUpTests(unittest.TestCase):
    def test_a_resident_cannot_speak_or_start_talking_on_the_turn_right_after_a_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = _ToolsLLM()
            sim = Simulation(_registry(tmp, llm, names=("Mara", "Finn")), EnvironmentAgent(seed=1), line_pacing=False)

            for _ in range(2):
                sim.tick()
                sim.wait_for_pending()

            # Mara's turns: the one that starts the chat, then the first one
            # after it, which has neither speaking nor starting another.
            first, after = llm.routine_tools["Mara"][:2]
            self.assertIn("initiate_conversation", first)
            self.assertFalse({SPEAK_TOOL_NAME, "initiate_conversation"} & after)

    def test_the_same_two_cannot_start_talking_again_right_away(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, _ChatLLM(), names=("Mara", "Finn"))
            sim = Simulation(registry, EnvironmentAgent(seed=1), line_pacing=False)
            sim.tick()
            sim.wait_for_pending()

            again = registry.get("Mara").tools.execute("initiate_conversation", {"target_name": "Finn"})
            self.assertIn("only just talked with Finn", again)

            sim._environment._elapsed_minutes += Simulation.REPEAT_CONVERSATION_MINUTES
            later = registry.get("Mara").tools.execute("initiate_conversation", {"target_name": "Finn"})
            self.assertIn("conversation with Finn", later)

    def test_a_turn_overtaken_by_a_conversation_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref: list = []
            registry = _registry(tmp, _OvertakenLLM(ref, ToolCall(name=SPEAK_TOOL_NAME, arguments={"text": "Hm."})))
            ref.append(registry)
            sim = Simulation(registry, EnvironmentAgent(seed=1), line_pacing=False)

            sim.tick()

            self.assertEqual(sim.state()["speech"], [])

    def test_an_overtaken_walk_is_not_queued_up_for_afterwards(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref: list = []
            registry = _registry(tmp, _OvertakenLLM(ref, ToolCall(name="move_to", arguments={"x": 0, "y": 0})))
            ref.append(registry)
            sim = Simulation(registry, EnvironmentAgent(seed=1), line_pacing=False)

            sim.tick()

            self.assertIsNone(registry.get("Gus").destination)

    def test_only_a_conversation_where_something_was_said_prompts_a_follow_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, _MutterLLM())
            sim = Simulation(registry, EnvironmentAgent(seed=1), line_pacing=False)
            from mini_map.simulation import _TravelerState

            st = sim._traveler_state["Finn"] = _TravelerState(exit_point=(100, 0), arrived_minute=0, last_turn_minute=0)
            silent = [ConversationTurn("Finn", "Mara", "", None, {"name": "wait", "arguments": {}, "result": ""})]
            sim._on_conversation_end(silent)
            self.assertIsNone(st.talked_with)

            spoken = [ConversationTurn("Finn", "Mara", "", "Hello.", None)]
            sim._on_conversation_end(spoken)
            self.assertEqual(st.talked_with, "Mara")


if __name__ == "__main__":
    unittest.main()
