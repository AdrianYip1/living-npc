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

from game_agents.agent import TurnResult
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

    def test_a_line_said_to_no_one_is_published_without_a_listener(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = Simulation(_registry(tmp, _MutterLLM(), names=("Mara",)), EnvironmentAgent(seed=1))

            sim.tick()

            [line] = sim.state()["speech"]
            self.assertEqual((line["speaker"], line["listener"], line["text"]), ("Mara", None, "Hm."))

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


class OverheardSpeechTests(unittest.TestCase):
    def test_a_line_said_aloud_reaches_whoever_is_in_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, _MutterLLM())  # Mara and Finn at (0, 0), Gus far off
            sim = Simulation(registry, EnvironmentAgent(seed=1), line_pacing=False)

            sim._record_turn(registry.get("Mara"), TurnResult(utterance="Morning, Finn.", action=None))

            [memory] = registry.get("Finn").memory.all()
            self.assertEqual(memory.content, 'Mara said aloud nearby: "Morning, Finn."')
            self.assertEqual(memory.tags, {"Mara"})
            self.assertEqual(registry.get("Gus").memory.all(), [])
            # ...and Mara remembers saying it, since someone heard -- so she
            # doesn't say the same thing again next turn.
            [own] = registry.get("Mara").memory.all()
            self.assertEqual(own.content, 'You said aloud, with Finn nearby: "Morning, Finn."')

    def test_a_line_nobody_hears_is_not_remembered(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(tmp, _MutterLLM(), names=("Mara", "Gus"))
            sim = Simulation(registry, EnvironmentAgent(seed=1), line_pacing=False)

            sim._record_turn(registry.get("Mara"), TurnResult(utterance="Hm.", action=None))

            self.assertEqual(registry.get("Mara").memory.all(), [])


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


class ConversationFollowUpTests(unittest.TestCase):
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
