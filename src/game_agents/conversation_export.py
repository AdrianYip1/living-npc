"""The conversation feed for the facial-animation/TTS program -- a separate
program that voices and animates NPC lines. Nothing in this codebase reads
these files back; the JSON shapes below are a contract with that program
and must not change without it changing too.

One conversation == one file, "{conversation_id}.jsonl", under the export
directory. Newline-delimited JSON, append-only: a consumer tailing the file
only ever sees new whole lines, never a changed one. Line shapes, told
apart by their keys:

  {"participants": [{"type", "id", "name"}, ...]}
      Written once, first. "type" is "npc" or "player"; the player's entry
      is {"type": "player", "id": "player", "name": "Player"}. An NPC's id
      is its name (names are unique in the registry).

  {"seq", "speaker_id", "phase", "gender", "text"}
      One per spoken NPC line. The player's own lines are never exported
      (only NPCs get voiced). `seq` counts up from 0. `phase` is "start" on
      the first line and "middle" after. `gender` is the speaker's, lower-
      cased. `text` is sanitized to what the speech side accepts -- see
      sanitize_for_speech().

      The end is its own final line of the same shape: "phase": "end",
      empty "text", seq = last seq + 1, echoing the last speaker's
      speaker_id/gender. With no lines at all it's
      {"seq": 0, "speaker_id": "", "phase": "end", "gender": "", "text": ""}.

Plus one pointer file, active.json, rewritten whole (atomically) whenever
it changes: {"conversation": "<file name>.jsonl" | null, "speakers": [ids]}
-- which conversation the renderer should show right now, and the ordered
NPC speaker ids, each mapping to a head slot. Choosing it is up to the
caller (see mini_map.Simulation); this module only writes it.
"""
from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

from .identity import Identity

ACTIVE_POINTER = "active.json"
PLAYER_ID = "player"
PLAYER_PARTICIPANT = {"type": "player", "id": PLAYER_ID, "name": "Player"}

# The speech side's parser accepts ASCII letters and digits, whitespace, and
# only , . ! ? ' " -- anything else breaks it.
_SMART_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"', "‟": '"'})
_DASH_RE = re.compile(r"\s*(?:-{2,}|[‒–—―⁃−])\s*")
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\s*\.\s*\.[.\s]*|…)\s*")
_DISALLOWED_CHAR_RE = re.compile(r"[^0-9A-Za-z\s,.!?'\"]")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?])")
_COMMA_THEN_PUNCT_RE = re.compile(r",(\s*[.!?])")
_REPEATED_PUNCT_RE = re.compile(r"([,.!?])\1+")
_LEADING_PUNCT_RE = re.compile(r"^[\s,.!?]+")
_TRAILING_COMMA_RE = re.compile(r"\s*,+\s*$")
_UNSAFE_FILENAME_RE = re.compile(r"[^0-9A-Za-z_-]+")


def sanitize_for_speech(text: str) -> str:
    """Forces a line into the speech side's character set. Accents fold to
    their base letter, smart quotes straighten, dashes/ellipses/colons/
    semicolons become commas, anything else disallowed becomes a space,
    then the leftover punctuation and spacing is tidied up.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.translate(_SMART_QUOTE_MAP)
    text = _DASH_RE.sub(", ", text)
    text = _ELLIPSIS_RE.sub(", ", text)
    text = text.replace(";", ",").replace(":", ",")
    text = _DISALLOWED_CHAR_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _COMMA_THEN_PUNCT_RE.sub(r"\1", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = _LEADING_PUNCT_RE.sub("", text)
    text = _TRAILING_COMMA_RE.sub(".", text.rstrip())
    return text.strip()


def npc_participant(identity: Identity) -> dict[str, str]:
    return {"type": "npc", "id": identity.name, "name": identity.name}


class ConversationExporter:
    """Writes the files described in the module docstring. Thread-safe:
    conversations run on worker threads and the player's on the HTTP one.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._lock = threading.Lock()
        # conversation_id -> the last line written (None before the first).
        self._open: dict[str, dict[str, Any] | None] = {}
        self._pointer: dict[str, Any] | None = None

    def clear(self) -> None:
        """Empties the directory (creating it if missing, so the other side
        can watch it from startup) -- nothing carries over between runs.
        """
        with self._lock:
            self._open.clear()
            self._pointer = None
            self.directory.mkdir(parents=True, exist_ok=True)
            for entry in self.directory.iterdir():
                if entry.is_file():
                    entry.unlink()

    def start(self, participants: list[dict[str, str]]) -> str:
        """Writes the participants line; returns the id to pass to line()/end()."""
        with self._lock:
            base = "conv_" + "_".join(_UNSAFE_FILENAME_RE.sub("-", p["id"]) for p in participants)
            conversation_id = f"{base}_{int(time.time() * 1000)}"
            n = 2
            while conversation_id in self._open or self._path(conversation_id).exists():
                conversation_id, n = f"{base}_{int(time.time() * 1000)}_{n}", n + 1
            self._open[conversation_id] = None
            self._append(conversation_id, {"participants": participants})
            return conversation_id

    def line(self, conversation_id: str, identity: Identity, text: str) -> None:
        """One spoken NPC line. Dropped if nothing speakable survives
        sanitizing -- an empty "text" isn't a line to the speech side.
        """
        text = sanitize_for_speech(text)
        if not text:
            return
        with self._lock:
            if conversation_id not in self._open:
                return
            last = self._open[conversation_id]
            line = {
                "seq": 0 if last is None else last["seq"] + 1,
                "speaker_id": identity.name,
                "phase": "start" if last is None else "middle",
                "gender": identity.gender.lower(),
                "text": text,
            }
            self._open[conversation_id] = line
            self._append(conversation_id, line)

    def end(self, conversation_id: str) -> None:
        """Writes the terminal line. Safe to call twice; the second is a no-op."""
        with self._lock:
            if conversation_id not in self._open:
                return
            last = self._open.pop(conversation_id)
            if last is None:
                line = {"seq": 0, "speaker_id": "", "phase": "end", "gender": "", "text": ""}
            else:
                line = {
                    "seq": last["seq"] + 1,
                    "speaker_id": last["speaker_id"],
                    "phase": "end",
                    "gender": last["gender"],
                    "text": "",
                }
            self._append(conversation_id, line)

    def is_open(self, conversation_id: str) -> bool:
        with self._lock:
            return conversation_id in self._open

    def point_at(self, conversation_id: str | None, speakers: list[str]) -> None:
        """Rewrites active.json if it changed. Replaced atomically so the
        reader never sees half a file; if the reader has it open right then
        (Windows refuses the replace), the next call tries again.
        """
        pointer = {
            "conversation": None if conversation_id is None else f"{conversation_id}.jsonl",
            "speakers": list(speakers) if conversation_id is not None else [],
        }
        with self._lock:
            if pointer == self._pointer:
                return
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                tmp = self.directory / f"{ACTIVE_POINTER}.tmp"
                tmp.write_text(json.dumps(pointer), encoding="utf-8")
                tmp.replace(self.directory / ACTIVE_POINTER)
            except OSError:
                return
            self._pointer = pointer

    def _path(self, conversation_id: str) -> Path:
        return self.directory / f"{conversation_id}.jsonl"

    def _append(self, conversation_id: str, obj: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._path(conversation_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(obj) + "\n")
