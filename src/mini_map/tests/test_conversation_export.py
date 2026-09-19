"""The Simulation's side of the conversation feed: every conversation gets
exported as it happens, and active.json points at the right one.
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from environment_agent.agent import EnvironmentAgent

from game_agents.conversation_export import ConversationExporter
from game_agents.identity import Identity
from game_agents.registry import NPCRegistry
from game_agents.storage import save_identities
from game_agents.tests.test_conversation_export import replay_like_renderer

from mini_map.simulation import Simulation
from mini_map.tests.test_npc_conversations import _ChatLLM, _MutterLLM


def _registry(tmp, llm) -> NPCRegistry:
    npcs_path = Path(tmp) / "npcs.json"
    save_identities(
        [
            Identity(name="Mara", traits=[], backstory="", speech_style="", gender="female"),
            Identity(name="Finn", traits=[], backstory="", speech_style="", gender="male"),
        ],
        npcs_path,
    )
    return NPCRegistry(npcs_path, Path(tmp) / "memory", llm, conversation_turns=10)


def _files(log: Path) -> list[Path]:
    return sorted(log.glob("conv_*.jsonl"))


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _pointer(log: Path) -> dict:
    return json.loads((log / "active.json").read_text(encoding="utf-8"))


class NpcConversationExportTests(unittest.TestCase):
    def test_an_npc_conversation_is_exported_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log"
            sim = Simulation(
                _registry(tmp, _ChatLLM()), EnvironmentAgent(seed=1), line_pacing=False, exporter=ConversationExporter(log)
            )

            sim.tick()
            sim.wait_for_pending()

            [path] = _files(log)
            lines = _read(path)
            self.assertEqual(
                lines[0],
                {"participants": [
                    {"type": "npc", "id": "Mara", "name": "Mara"},
                    {"type": "npc", "id": "Finn", "name": "Finn"},
                ]},
            )
            self.assertEqual(
                [(l["seq"], l["speaker_id"], l["phase"], l["gender"], l["text"]) for l in lines[1:]],
                [
                    (0, "Mara", "start", "female", "Mara line 1"),
                    (1, "Finn", "middle", "male", "Finn line 1"),
                    (2, "Mara", "middle", "female", "Mara line 2"),
                    (3, "Finn", "middle", "male", "Finn line 2"),
                    (4, "Mara", "middle", "female", "Mara line 3"),
                    (5, "Mara", "end", "female", ""),
                ],
            )

    def test_pointer_follows_the_conversation_while_it_lasts(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log"
            gate = threading.Event()
            sim = Simulation(
                _registry(tmp, _ChatLLM(gate)),
                EnvironmentAgent(seed=1),
                line_pacing=False,
                exporter=ConversationExporter(log),
            )

            sim.tick()  # the exchange is held open on the gate
            deadline = time.monotonic() + 5
            sim.update_active_conversation()
            while _pointer(log)["conversation"] is None and time.monotonic() < deadline:  # starts on a worker thread
                time.sleep(0.01)
                sim.update_active_conversation()
            [path] = _files(log)
            self.assertEqual(_pointer(log), {"conversation": path.name, "speakers": ["Mara", "Finn"], "seq": 0})

            gate.set()
            sim.wait_for_pending()
            self.assertEqual(
                replay_like_renderer(log),
                [(0, "female", "Mara line 1"), (1, "male", "Finn line 1"), (0, "female", "Mara line 2"),
                 (1, "male", "Finn line 2"), (0, "female", "Mara line 3")],
            )
            sim.update_active_conversation()
            self.assertEqual(_pointer(log), {"conversation": None, "speakers": []})


class PlayerConversationExportTests(unittest.TestCase):
    def test_player_conversation_exports_only_npc_lines_and_wins_the_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log"
            sim = Simulation(
                _registry(tmp, _MutterLLM()), EnvironmentAgent(seed=1), line_pacing=False, exporter=ConversationExporter(log)
            )

            self.assertTrue(sim.start_conversation("Finn"))
            sim.say("Finn", "hello there")
            sim.update_active_conversation()
            [path] = _files(log)
            self.assertEqual(_pointer(log), {"conversation": path.name, "speakers": ["Finn"], "seq": 0})

            sim.end_conversation("Finn")
            sim.say("Finn", "one more thing")  # after leaving: not exported
            self.assertEqual(
                _read(path),
                [
                    {"participants": [
                        {"type": "player", "id": "player", "name": "Player"},
                        {"type": "npc", "id": "Finn", "name": "Finn"},
                    ]},
                    {"seq": 0, "speaker_id": "Finn", "phase": "start", "gender": "male", "text": "Hm."},
                    {"seq": 1, "speaker_id": "Finn", "phase": "end", "gender": "male", "text": ""},
                ],
            )
            sim.update_active_conversation()
            self.assertEqual(_pointer(log), {"conversation": None, "speakers": []})


class NpcStateExportTests(unittest.TestCase):
    def test_every_npc_gets_a_slot_and_keeps_facing_its_last_movement(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log"
            registry = _registry(tmp, _MutterLLM())
            sim = Simulation(registry, EnvironmentAgent(seed=1), line_pacing=False, exporter=ConversationExporter(log))
            mara, finn = registry.get("Mara"), registry.get("Finn")
            mara.position, finn.position = (-100.0, 0.0), (100.0, 100.0)
            mara.velocity = (-5.0, 0.0)
            sim.export_npc_state()
            mara.velocity = (0.0, 0.0)  # stopped: still faces -x
            sim.export_npc_state()
            state = json.loads((log / "npc_state.json").read_text(encoding="utf-8"))
            self.assertEqual(
                state,
                {"npcs": [
                    {"slot": 0, "x": 0.0, "z": 0.5, "rot": -1.571},
                    {"slot": 1, "x": 1.0, "z": 1.0, "rot": 0.0},
                ]},
            )


class SpeechPacingTests(unittest.TestCase):
    """While the speech side voices a conversation, a line waits for the
    previous one to be spoken rather than for its reading time.
    """

    def _sim(self, tmp) -> Simulation:
        return Simulation(_registry(tmp, _MutterLLM()), EnvironmentAgent(seed=1), exporter=ConversationExporter(Path(tmp) / "log"))

    def _time(self, sim: Simulation, answers: list[bool | None], **kwargs) -> float:
        start = time.monotonic()
        sim._hold_until(start + 0.3, spoken=lambda: answers.pop(0) if len(answers) > 1 else answers[0], **kwargs)
        return time.monotonic() - start

    def test_spoken_ends_the_wait_before_the_reading_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertLess(self._time(self._sim(tmp), [True], speech_deadline=time.monotonic() + 5), 0.2)

    def test_not_yet_spoken_waits_past_the_reading_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            elapsed = self._time(self._sim(tmp), [False] * 12 + [True], speech_deadline=time.monotonic() + 5)
            self.assertGreater(elapsed, 0.5)
            self.assertLess(elapsed, 2)

    def test_not_yet_spoken_gives_up_at_the_speech_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            elapsed = self._time(self._sim(tmp), [False], speech_deadline=time.monotonic() + 0.6)
            self.assertGreater(elapsed, 0.5)
            self.assertLess(elapsed, 2)

    def test_nobody_voicing_it_falls_back_to_the_reading_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            elapsed = self._time(self._sim(tmp), [None], speech_deadline=time.monotonic() + 5)
            self.assertGreater(elapsed, 0.25)
            self.assertLess(elapsed, 1)

    def test_a_voiced_conversation_is_paced_by_the_speech_side(self):
        """A fake speech side voices each line in 0.1 s. Reading time is
        made huge, so the exchange only finishes quickly if the acks drive it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log"
            sim = Simulation(_registry(tmp, _ChatLLM()), EnvironmentAgent(seed=1), exporter=ConversationExporter(log))
            sim.MIN_LINE_SECONDS = sim.MAX_LINE_SECONDS = 30.0
            voiced: list[tuple[str, int]] = []
            stop = threading.Event()

            def speech_side() -> None:
                ack = ("", -1)
                while not stop.is_set():
                    sim.update_active_conversation()
                    pointer = _pointer(log) if (log / "active.json").exists() else {}
                    name = pointer.get("conversation")
                    if name:
                        for line in _read(log / name)[1:]:
                            if line["text"] and (name, line["seq"]) not in voiced and line["seq"] >= pointer["seq"]:
                                time.sleep(0.1)
                                voiced.append((name, line["seq"]))
                                ack = (name, line["seq"])
                                break
                    tmp_ack = log / "spoken.json.tmp"
                    tmp_ack.write_text(json.dumps({"conversation": ack[0], "seq": ack[1]}), encoding="utf-8")
                    tmp_ack.replace(log / "spoken.json")
                    time.sleep(0.02)

            log.mkdir(parents=True)
            worker = threading.Thread(target=speech_side, daemon=True)
            worker.start()
            try:
                start = time.monotonic()
                sim.tick()
                sim.wait_for_pending()
                elapsed = time.monotonic() - start
            finally:
                stop.set()
                worker.join()
            self.assertLess(elapsed, 10)
            self.assertEqual([seq for _, seq in voiced], [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
