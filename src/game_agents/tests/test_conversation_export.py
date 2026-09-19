"""The conversation feed's file format is a contract with the speech/
animation program -- these pin its exact shape.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from game_agents.conversation_export import (
    PLAYER_PARTICIPANT,
    ConversationExporter,
    npc_participant,
    sanitize_for_speech,
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
                {"conversation": "conv_Mara_Finn_1.jsonl", "speakers": ["Mara", "Finn"]},
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
