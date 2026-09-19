"""Every name a traveler has gone by, across runs. Residents' memories
persist between runs, so a name reused in a later run reads as the same
person back again: Mara greets a brand-new Osric as a regular. The
simulation keeps new travelers off these names (see
Simulation._admit_traveler).
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Iterable

from .agent import Agent

# Numbered duplicates from before travelers stopped being numbered
# ("Odell 2") -- the name is still "Odell".
_NUMBER_SUFFIX = re.compile(r"\s+\d+$")


def names_in_memories(residents: Iterable[Agent]) -> list[str]:
    """Everyone the residents remember meeting, other than each other and
    the player, oldest last-mention first. Memories are tagged with the
    other person's name (see NPCRegistry), so this covers runs from before
    the used-names file existed.
    """
    residents = list(residents)
    skip = {agent.identity.name.lower() for agent in residents} | {"player"}
    last_seen: dict[str, float] = {}
    for agent in residents:
        for memory in agent.memory.all():
            for tag in memory.tags:
                name = _NUMBER_SUFFIX.sub("", tag).strip()
                if name and name.lower() not in skip:
                    last_seen[name] = max(last_seen.get(name, 0.0), memory.timestamp)
    return sorted(last_seen, key=last_seen.__getitem__)


class UsedTravelerNames:
    """The names, least recently used first, saved to `path` (if given) on
    every change. `remembered` names (see names_in_memories) are merged in
    as older than anything in the file.
    """

    def __init__(self, path: str | Path | None = None, *, remembered: Iterable[str] = ()) -> None:
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        saved: list[str] = []
        if self._path is not None and self._path.exists():
            saved = [name for name in json.loads(self._path.read_text(encoding="utf-8")) if isinstance(name, str)]
        seen = {name.lower() for name in saved}
        self._names = [name for name in dict.fromkeys(remembered) if name.lower() not in seen] + saved

    def names(self) -> list[str]:
        with self._lock:
            return list(self._names)

    def record(self, name: str) -> None:
        """Marks `name` as just used: moved to the end, and saved."""
        with self._lock:
            self._names = [n for n in self._names if n.lower() != name.lower()] + [name]
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(json.dumps(self._names, indent=2), encoding="utf-8")
