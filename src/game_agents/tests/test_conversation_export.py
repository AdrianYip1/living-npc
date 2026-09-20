"""The conversation feed's file format is a contract with the speech/
animation program -- these pin its exact shape.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
import unittest
from pathlib import Path

from game_agents.conversation_export import (
    PLAYER_PARTICIPANT,
    ConversationExporter,
    npc_participant,
    npc_state_entry,
    player_from_renderer,
    sanitize_for_speech,
    time_state,
)
from game_agents.identity import Identity


def _identity(name: str, gender: str) -> Identity:
    return Identity(name=name, traits=[], backstory="", speech_style="", gender=gender)


MARA = _identity("Mara", "Female")
FINN = _identity("Finn", "male")


def _lines(directory: Path, conversation_id: str) -> list[dict]:
    text = (directory / f"{conversation_id}.jsonl").read_text(encoding="utf-8")
    assert text.endswith("\n")
    return [json.loads(line) for line in text.splitlines()]


def replay_like_renderer(directory: Path) -> list[tuple[int, str, str]]:
    """What the renderer's parser would voice, polled until it stops
    producing lines: it reads active.json, skips to the first line with
    seq >= the next one it wants, stops at an "end" line, and maps the
    speaker to a head slot by its position in "speakers".
    """
    spoken: list[tuple[int, str, str]] = []
    turn_index = 0
    while True:
        pointer = json.loads((directory / "active.json").read_text(encoding="utf-8"))
        if not pointer.get("conversation"):
            return spoken
        speakers = pointer.get("speakers", [])
        next_line = None
        for raw in (directory / pointer["conversation"]).read_text(encoding="utf-8").splitlines():
            obj = json.loads(raw)
            if obj.get("phase") == "end":
                break
            if obj.get("seq", -1) >= turn_index:
                next_line = obj
                break
        if next_line is None:
            return spoken
        turn_index = next_line["seq"] + 1
        if next_line["speaker_id"] in speakers and next_line["text"]:
            spoken.append((speakers.index(next_line["speaker_id"]), next_line["gender"], next_line["text"]))


class FormatTests(unittest.TestCase):
    def test_npc_conversation_file_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter = ConversationExporter(tmp)
            cid = exporter.start([npc_participant(MARA), npc_participant(FINN)])
            exporter.line(cid, MARA, "Morning.")
            exporter.line(cid, FINN, "Morning, Mara!")
            exporter.line(cid, MARA, "Bye.")
            exporter.end(cid)

            self.assertTrue(cid.startswith("conv_Mara_Finn_"))
            self.assertEqual(
                _lines(Path(tmp), cid),
                [
                    {"participants": [
                        {"type": "npc", "id": "Mara", "name": "Mara"},
                        {"type": "npc", "id": "Finn", "name": "Finn"},
                    ]},
                    {"seq": 0, "speaker_id": "Mara", "phase": "start", "gender": "female", "text": "Morning."},
                    {"seq": 1, "speaker_id": "Finn", "phase": "middle", "gender": "male", "text": "Morning, Mara!"},
                    {"seq": 2, "speaker_id": "Mara", "phase": "middle", "gender": "female", "text": "Bye."},
                    {"seq": 3, "speaker_id": "Mara", "phase": "end", "gender": "female", "text": ""},
                ],
            )

    def test_key_order_matches_the_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter = ConversationExporter(tmp)
            cid = exporter.start([npc_participant(MARA)])
            exporter.line(cid, MARA, "Hi.")
            raw = (Path(tmp) / f"{cid}.jsonl").read_text(encoding="utf-8").splitlines()[1]
            self.assertEqual(list(json.loads(raw)), ["seq", "speaker_id", "phase", "gender", "text"])

    def test_player_participant_and_empty_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter = ConversationExporter(tmp)
            cid = exporter.start([dict(PLAYER_PARTICIPANT), npc_participant(FINN)])
            exporter.end(cid)
            exporter.end(cid)  # second end is a no-op
            self.assertTrue(cid.startswith("conv_player_Finn_"))
            self.assertEqual(
                _lines(Path(tmp), cid),
                [
                    {"participants": [
                        {"type": "player", "id": "player", "name": "Player"},
                        {"type": "npc", "id": "Finn", "name": "Finn"},
                    ]},
                    {"seq": 0, "speaker_id": "", "phase": "end", "gender": "", "text": ""},
                ],
            )

    def test_lines_after_end_and_unspeakable_lines_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter = ConversationExporter(tmp)
            cid = exporter.start([npc_participant(MARA)])
            exporter.line(cid, MARA, "... :) —")
            exporter.end(cid)
            exporter.line(cid, MARA, "Too late.")
            self.assertEqual(len(_lines(Path(tmp), cid)), 2)

    def test_names_with_spaces_make_safe_file_names_but_keep_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter = ConversationExporter(tmp)
            brynn = _identity("Brynn 2", "female")
            cid = exporter.start([npc_participant(brynn)])
            self.assertTrue(cid.startswith("conv_Brynn-2_"))
            self.assertEqual(_lines(Path(tmp), cid)[0]["participants"][0]["id"], "Brynn 2")


class PointerTests(unittest.TestCase):
    def test_pointer_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter = ConversationExporter(tmp)
            exporter.point_at("conv_Mara_Finn_1", ["Mara", "Finn"])
            self.assertEqual(
                json.loads((Path(tmp) / "active.json").read_text(encoding="utf-8")),
                {"conversation": "conv_Mara_Finn_1.jsonl", "speakers": ["Mara", "Finn"], "seq": 0},
            )
            exporter.point_at(None, ["ignored"])
            self.assertEqual(
                json.loads((Path(tmp) / "active.json").read_text(encoding="utf-8")),
                {"conversation": None, "speakers": []},
            )

    def test_clear_empties_the_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter = ConversationExporter(Path(tmp) / "log")
            cid = exporter.start([npc_participant(MARA)])
            exporter.point_at(cid, ["Mara"])
            exporter.clear()
            self.assertEqual(list((Path(tmp) / "log").iterdir()), [])

    def test_renderer_replays_every_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter = ConversationExporter(tmp)
            cid = exporter.start([npc_participant(MARA), npc_participant(FINN)])
            exporter.point_at(cid, ["Mara", "Finn"])
            exporter.line(cid, MARA, "Morning.")
            exporter.line(cid, FINN, "Café's open—come in…")
            exporter.end(cid)
            self.assertEqual(
                replay_like_renderer(Path(tmp)),
                [(0, "female", "Morning."), (1, "male", "Cafe's open, come in.")],
            )


class PointerSeqTests(unittest.TestCase):
    def test_pointer_moving_to_a_conversation_in_progress_starts_at_its_latest_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter = ConversationExporter(tmp)
            cid = exporter.start([npc_participant(MARA), npc_participant(FINN)])
            exporter.line(cid, MARA, "Morning.")
            exporter.line(cid, FINN, "Morning, Mara!")
            exporter.point_at(cid, ["Mara", "Finn"])
            self.assertEqual(_pointer(Path(tmp))["seq"], 1)

    def test_a_new_line_in_the_same_conversation_does_not_move_the_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter = ConversationExporter(tmp)
            cid = exporter.start([npc_participant(MARA), npc_participant(FINN)])
            exporter.point_at(cid, ["Mara", "Finn"])
            exporter.line(cid, MARA, "Morning.")
            exporter.point_at(cid, ["Mara", "Finn"])
            self.assertEqual(_pointer(Path(tmp))["seq"], 0)

    def test_line_returns_its_seq(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter = ConversationExporter(tmp)
            cid = exporter.start([npc_participant(MARA)])
            self.assertEqual(exporter.line(cid, MARA, "Morning."), 0)
            self.assertIsNone(exporter.line(cid, MARA, "..."))  # nothing speakable
            self.assertEqual(exporter.line(cid, MARA, "Bye."), 1)


def _pointer(directory: Path) -> dict:
    return json.loads((directory / "active.json").read_text(encoding="utf-8"))


def _ack(directory: Path, conversation: str, seq: int, age: float = 0.0) -> None:
    """spoken.json as the speech side writes it, `age` seconds old."""
    path = directory / "spoken.json"
    path.write_text(json.dumps({"conversation": conversation, "seq": seq}), encoding="utf-8")
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))


class SpeechDoneTests(unittest.TestCase):
    def _voiced(self, tmp: str) -> tuple[ConversationExporter, str]:
        exporter = ConversationExporter(tmp)
        cid = exporter.start([npc_participant(MARA), npc_participant(FINN)])
        exporter.line(cid, MARA, "Morning.")
        exporter.line(cid, FINN, "Morning, Mara!")
        exporter.point_at(cid, ["Mara", "Finn"])
        return exporter, cid

    def test_waits_until_the_ack_reaches_the_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter, cid = self._voiced(tmp)
            _ack(Path(tmp), f"{cid}.jsonl", 0)
            self.assertIs(exporter.speech_done(cid, 1), False)
            _ack(Path(tmp), f"{cid}.jsonl", 1)
            self.assertIs(exporter.speech_done(cid, 1), True)

    def test_an_ack_for_another_conversation_is_not_this_one_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter, cid = self._voiced(tmp)
            _ack(Path(tmp), "conv_other_1.jsonl", 99)
            self.assertIs(exporter.speech_done(cid, 1), False)

    def test_the_initial_heartbeat_is_not_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter, cid = self._voiced(tmp)
            _ack(Path(tmp), "", -1)
            self.assertIs(exporter.speech_done(cid, 1), False)

    def test_none_when_nobody_is_voicing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter, cid = self._voiced(tmp)
            self.assertIsNone(exporter.speech_done(cid, 1))  # no spoken.json yet
            _ack(Path(tmp), f"{cid}.jsonl", 0, age=5.0)
            self.assertIsNone(exporter.speech_done(cid, 1))  # speech side stopped
            _ack(Path(tmp), f"{cid}.jsonl", 0)
            exporter.point_at(None, [])
            self.assertIsNone(exporter.speech_done(cid, 1))  # pointer moved away


class NpcStateTests(unittest.TestCase):
    def test_entry_normalizes_the_map_and_faces_the_last_movement(self):
        self.assertEqual(
            npc_state_entry(0, (-100, 100), (0.0, 1.0), name="Mara"),
            {"slot": 0, "name": "Mara", "x": 0.0, "z": 1.0, "rot": 0.0},
        )
        self.assertEqual(
            npc_state_entry(1, (0, 50), (3.0, 0.0)),
            {"slot": 1, "name": "", "x": 0.5, "z": 0.75, "rot": 1.571},
        )
        self.assertEqual(npc_state_entry(2, (500, -500), (0.0, -1.0))["x"], 1.0)  # clamped

    def test_time_state(self):
        self.assertEqual(
            time_state(720, "afternoon", 2.0),
            {"minute": 720, "day_fraction": 0.5, "phase": "afternoon", "rate": 2.0},
        )
        self.assertEqual(time_state(495, "morning", 0.0)["day_fraction"], 0.3438)

    def test_written_whole_and_only_when_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter = ConversationExporter(tmp)
            npcs = [npc_state_entry(0, (0, 0), (0.0, 1.0))]
            noon = time_state(720, "afternoon", 2.0)
            exporter.write_npc_state(noon, npcs)
            path = Path(tmp) / "npc_state.json"
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"time": noon, "npcs": npcs})
            path.unlink()
            exporter.write_npc_state(dict(noon), list(npcs))
            self.assertFalse(path.exists())

    def test_rewritten_when_only_the_time_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter = ConversationExporter(tmp)
            npcs = [npc_state_entry(0, (0, 0), (0.0, 1.0))]
            exporter.write_npc_state(time_state(720, "afternoon", 2.0), npcs)
            exporter.write_npc_state(time_state(721, "afternoon", 2.0), npcs)
            path = Path(tmp) / "npc_state.json"
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["time"]["minute"], 721)

    def test_clear_keeps_the_renderers_own_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "bounds.json").write_text("{}", encoding="utf-8")
            (directory / "spoken.json").write_text("{}", encoding="utf-8")
            (directory / "player_state.json").write_text("{}", encoding="utf-8")
            exporter = ConversationExporter(directory)
            exporter.write_npc_state(time_state(0, "night", 0.0), [npc_state_entry(0, (0, 0), (0.0, 1.0))])
            exporter.point_at(exporter.start([npc_participant(MARA)]), ["Mara"])
            exporter.clear()
            self.assertEqual(
                sorted(entry.name for entry in directory.iterdir()),
                ["bounds.json", "player_state.json", "spoken.json"],
            )


def _player_state(directory: Path, x: float, z: float, rot: float = 0.0, age: float = 0.0) -> None:
    """player_state.json as the renderer writes it, `age` seconds old."""
    path = directory / "player_state.json"
    path.write_text(json.dumps({"x": x, "z": z, "rot": rot}), encoding="utf-8")
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))


class PlayerStateTests(unittest.TestCase):
    """player_state.json, the renderer's half of the position contract:
    where the person in the 3D scene is, coming back as minimap coordinates.
    """

    def test_it_undoes_npc_state_entry(self):
        for position, facing in (((-100, 100), (0.0, 1.0)), ((0, 50), (3.0, 0.0)), ((42, -17), (-2.0, -5.0))):
            entry = npc_state_entry(0, position, facing)
            placed, angle = player_from_renderer(entry)
            self.assertAlmostEqual(placed[0], position[0], places=1)
            self.assertAlmostEqual(placed[1], position[1], places=1)
            # The same direction the entry faced, as an angle from +x.
            self.assertAlmostEqual(math.cos(angle), facing[0] / math.hypot(*facing), places=2)
            self.assertAlmostEqual(math.sin(angle), facing[1] / math.hypot(*facing), places=2)

    def test_a_position_off_the_map_is_clamped_to_its_edge(self):
        self.assertEqual(player_from_renderer({"x": 2.0, "z": -1.0})[0], (100.0, -100.0))

    def test_nothing_comes_of_a_file_that_is_not_the_contract(self):
        for entry in ({}, {"x": 0.5}, {"x": "left", "z": 0.5}, {"x": 0.5, "z": float("nan")}, [], None):
            self.assertIsNone(player_from_renderer(entry))

    def test_read_ignores_a_missing_or_stale_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            exporter = ConversationExporter(directory)
            self.assertIsNone(exporter.read_player_state())  # renderer not running
            _player_state(directory, 0.75, 0.25, age=5.0)
            self.assertIsNone(exporter.read_player_state())  # renderer stopped
            _player_state(directory, 0.75, 0.25)
            self.assertEqual(exporter.read_player_state()[0], (50.0, -50.0))


class SanitizeTests(unittest.TestCase):
    def test_only_the_allowed_characters_survive(self):
        cases = {
            "Naïve café “owner” — yes; no: maybe…": 'Naive cafe "owner", yes, no, maybe.',
            "Wait!!! (really?) #1 & 2": "Wait! really? 1 2",
            "Hello, ...": "Hello.",
            "…well": "well",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_for_speech(raw), expected)

    def test_output_is_always_within_the_allowed_set(self):
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n,.!?'\"")
        text = sanitize_for_speech("¿Qué? <b>bold</b> *wave* 50% off — «deal» ✓ 日本")
        self.assertTrue(set(text) <= allowed, text)


if __name__ == "__main__":
    unittest.main()
