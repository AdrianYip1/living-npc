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
it changes: {"conversation": "<file name>.jsonl", "speakers": [ids], "seq"}
or {"conversation": null, "speakers": []} -- which conversation the
renderer should show right now, and the ordered NPC speaker ids, each
mapping to a head slot. "seq" is the conversation's latest line when the
pointer moved to it: the renderer starts voicing there, so walking up to
a conversation in progress plays from "now" instead of replaying it.
Choosing it is up to the caller (see mini_map.Simulation); this module
only writes it.

The one file flowing the other way is spoken.json, written by the speech
side: {"conversation": "<file name>.jsonl", "seq"} -- the last line it
finished voicing. It rewrites that every ~250 ms even when nothing has
changed, so its age doubles as a heartbeat: a stale file means the speech
side isn't running. speech_done() reads it, so the simulation can hold a
conversation's next line until the previous one has actually been heard.

npc_state.json places NPC bodies in the 3D scene and carries the in-game
time, rewritten whole (atomically) as either changes:
{"time": {"minute", "day_fraction", "phase", "rate"},
 "npcs": [{"slot", "x", "z", "rot"}, ...]}.
"minute" is the minute of the day, 0..1439; "day_fraction" is minute / 1440
(0 = midnight, 0.5 = noon); "phase" is "night" (00-06), "morning",
"afternoon" or "evening" (18-24). The clock only moves in whole-tick steps,
so "rate" -- in-game minutes per real second, 0 while paused -- lets the
renderer advance the time smoothly between rewrites.
"slot" indexes the renderer's fixed list of bodies -- it ignores slots it
has no body for. "x"/"z" are the minimap position normalized to 0..1
across the map (minimap y -> scene z), which the renderer stretches over
its world bounds. "rot" is the facing about the up axis, in radians,
atan2(vx, vy) of the last movement. PLACEHOLDER: which NPC gets which slot
(residents first, in registry order, then travelers) and the axis/facing
conventions are unsettled with the renderer side.

player_state.json is npc_state.json's mirror, written by the renderer:
{"x", "z", "rot"} -- the same shape as one of its "npcs" entries without
the slot. Where the person walking around the 3D scene is, normalized
0..1 across the renderer's own bounds, and their facing about the up axis.
It's rewritten every ~250 ms even when nothing moved, so like spoken.json
its age doubles as a heartbeat. read_player_state() turns a fresh one back
into minimap coordinates: that is how the 3D scene moves the player's dot
on the 2D map, and while it keeps coming the page's own reported position
is ignored (see mini_map.Simulation.set_player_position).

The renderer also writes bounds.json once at its startup ({"minX", "maxX",
"minZ", "maxZ"}, its scene's footprint in world units). Nothing here reads
it yet, but clear() leaves it -- and spoken.json -- alone: they belong to
the other side, which may have started first.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

from .identity import Identity
from .world import MAP_MAX, MAP_MIN

ACTIVE_POINTER = "active.json"
SPOKEN_ACK = "spoken.json"
NPC_STATE = "npc_state.json"
PLAYER_STATE = "player_state.json"
RENDERER_BOUNDS = "bounds.json"
# Written by the renderer, not us -- clear() must leave these be.
RENDERER_OWNED = frozenset({SPOKEN_ACK, RENDERER_BOUNDS, PLAYER_STATE})
# How old spoken.json may be before the speech side counts as not running
# -- it rewrites the file about every 250 ms. player_state.json comes on
# the same beat, so it gets the same allowance.
SPEECH_SIDE_STALE_SECONDS = 2.0
RENDERER_SIDE_STALE_SECONDS = 2.0
PLAYER_ID = "player"
PLAYER_PARTICIPANT = {"type": "player", "id": PLAYER_ID, "name": "Player"}
_MINUTES_PER_DAY = 24 * 60

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


def npc_state_entry(slot: int, position: tuple[float, float], facing: tuple[float, float], name: str = "") -> dict[str, Any]:
    """One npc_state.json entry: a minimap position and the direction the
    NPC last moved in, converted to the renderer's terms (see the module
    docstring). Rounded so jitter too small to see doesn't rewrite the file.
    """
    span = MAP_MAX - MAP_MIN
    return {
        "slot": slot,
        "name": name,
        "x": round(min(1.0, max(0.0, (position[0] - MAP_MIN) / span)), 4),
        "z": round(min(1.0, max(0.0, (position[1] - MAP_MIN) / span)), 4),
        "rot": round(math.atan2(facing[0], facing[1]), 3),
    }


def player_from_renderer(entry: dict[str, Any]) -> tuple[tuple[float, float], float] | None:
    """npc_state_entry() backwards, for the renderer's player_state.json:
    its normalized {"x", "z"} back to a minimap position, and its "rot"
    back to the minimap's own facing angle (0 = +x, turning toward +y).
    None if the file isn't the shape the contract says -- caught
    half-written, or a renderer sending something else.
    """
    if not isinstance(entry, dict):
        return None
    try:
        nx = float(entry["x"])
        nz = float(entry["z"])
        rot = float(entry.get("rot", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(map(math.isfinite, (nx, nz, rot))):
        return None
    span = MAP_MAX - MAP_MIN
    position = (
        MAP_MIN + min(1.0, max(0.0, nx)) * span,
        MAP_MIN + min(1.0, max(0.0, nz)) * span,
    )
    # "rot" is atan2(vx, vy) of the movement (see npc_state_entry), so the
    # direction it names is (sin rot, cos rot) -- and the minimap wants
    # that as an angle from +x.
    return position, math.atan2(math.cos(rot), math.sin(rot))


def time_state(minute_of_day: int, phase: str, game_minutes_per_real_second: float) -> dict[str, Any]:
    """npc_state.json's "time" object (see the module docstring)."""
    return {
        "minute": minute_of_day,
        "day_fraction": round(minute_of_day / _MINUTES_PER_DAY, 4),
        "phase": phase,
        "rate": round(game_minutes_per_real_second, 4),
    }


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
        self._npc_state: dict[str, Any] | None = None

    def clear(self) -> None:
        """Empties the directory (creating it if missing, so the other side
        can watch it from startup) -- nothing of ours carries over between
        runs. The renderer's own files (RENDERER_OWNED) are kept.
        """
        with self._lock:
            self._open.clear()
            self._pointer = None
            self._npc_state = None
            self.directory.mkdir(parents=True, exist_ok=True)
            for entry in self.directory.iterdir():
                if entry.is_file() and entry.name not in RENDERER_OWNED:
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

    def line(self, conversation_id: str, identity: Identity, text: str) -> int | None:
        """One spoken NPC line; returns its seq. Dropped (returning None) if
        nothing speakable survives sanitizing -- an empty "text" isn't a
        line to the speech side.
        """
        text = sanitize_for_speech(text)
        if not text:
            return None
        with self._lock:
            if conversation_id not in self._open:
                return None
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
            return line["seq"]

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
        with self._lock:
            if conversation_id is None:
                pointer: dict[str, Any] = {"conversation": None, "speakers": []}
            else:
                pointer = {"conversation": f"{conversation_id}.jsonl", "speakers": list(speakers)}
            # "seq" only matters when the pointer moves, so a new line in the
            # same conversation isn't a change worth rewriting for.
            if self._pointer is not None and all(pointer[k] == self._pointer[k] for k in pointer):
                return
            if conversation_id is not None:
                last = self._open.get(conversation_id)
                pointer["seq"] = 0 if last is None else last["seq"]
            if self._replace(ACTIVE_POINTER, pointer):
                self._pointer = pointer

    def point_at_all(self, conversations: list[tuple[str, list[str]]]) -> None:
        """Writes all_active.json with every ongoing conversation so the
        renderer can animate and play audio for all of them in parallel.
        """
        with self._lock:
            entries = []
            for cid, speakers in conversations:
                last = self._open.get(cid)
                entries.append({
                    "conversation": f"{cid}.jsonl",
                    "speakers": list(speakers),
                    "seq": 0 if last is None else last["seq"],
                })
            self._replace("all_active.json", entries)

    def write_npc_state(self, time_now: dict[str, Any], npcs: list[dict[str, Any]]) -> None:
        """Rewrites npc_state.json if it changed. `time_now` comes from
        time_state(); `npcs` are {"slot", "x", "z", "rot"} entries -- see
        npc_state_entry(). Same atomic replace as point_at(), retried on the
        next call if refused.
        """
        state = {"time": time_now, "npcs": npcs}
        with self._lock:
            if state == self._npc_state:
                return
            if self._replace(NPC_STATE, state):
                self._npc_state = state

    def speech_done(self, conversation_id: str, seq: int) -> bool | None:
        """Whether the speech side has finished voicing line `seq` of this
        conversation. None if it isn't voicing this conversation at all --
        active.json points elsewhere, or spoken.json is missing or stale
        (the speech side isn't running) -- so the caller paces some other way.
        """
        with self._lock:
            pointer = self._pointer
        name = f"{conversation_id}.jsonl"
        if pointer is None or pointer["conversation"] != name:
            return None
        path = self.directory / SPOKEN_ACK
        try:
            if time.time() - path.stat().st_mtime > SPEECH_SIDE_STALE_SECONDS:
                return None
        except OSError:
            return None
        try:
            ack = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Caught mid-replace (Windows can refuse the read then) -- the
            # speech side is alive, so just not done yet; ask again.
            return False
        if not isinstance(ack, dict):
            return False
        acked = ack.get("seq")
        return ack.get("conversation") == name and isinstance(acked, int) and acked >= seq

    def read_player_state(self) -> tuple[tuple[float, float], float] | None:
        """Where the renderer has the player, as a minimap position and
        facing (see player_from_renderer). None if the renderer isn't
        running -- no player_state.json, or one too old to trust (it
        rewrites it about every 250 ms even standing still) -- so the
        caller can fall back to the position the page reports.
        """
        path = self.directory / PLAYER_STATE
        try:
            if time.time() - path.stat().st_mtime > RENDERER_SIDE_STALE_SECONDS:
                return None
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Missing, or caught mid-replace -- ask again next beat.
            return None
        return player_from_renderer(entry)

    def _replace(self, name: str, obj: Any) -> bool:
        """Writes a whole file via a temp file and rename, so the reader
        never sees half of it. False if that failed -- e.g. the reader had
        it open right then, which Windows refuses to replace.
        """
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            tmp = self.directory / f"{name}.tmp"
            tmp.write_text(json.dumps(obj), encoding="utf-8")
            tmp.replace(self.directory / name)
        except OSError:
            return False
        return True

    def _path(self, conversation_id: str) -> Path:
        return self.directory / f"{conversation_id}.jsonl"

    def _append(self, conversation_id: str, obj: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._path(conversation_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(obj) + "\n")
