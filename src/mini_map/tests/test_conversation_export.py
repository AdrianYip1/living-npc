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
            self.assertEqual(_pointer(log), {"conversation": path.name, "speakers": ["Mara", "Finn"]})

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
            self.assertEqual(_pointer(log), {"conversation": path.name, "speakers": ["Finn"]})

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


if __name__ == "__main__":
    unittest.main()
