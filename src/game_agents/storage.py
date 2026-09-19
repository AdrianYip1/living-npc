from __future__ import annotations

import json
from pathlib import Path

from .identity import Identity
from .inventory import Inventory
from .memory import MemoryStore

_IDENTITY_FIELDS = ("name", "traits", "backstory", "speech_style", "goals", "home", "workplace", "habits", "starting_money", "starting_items")
_COORD_FIELDS = ("home", "workplace")


def load_identities(path: str | Path) -> list[Identity]:
    p = Path(path)
    if not p.exists():
        return []
    records = json.loads(p.read_text(encoding="utf-8"))
    identities = []
    for record in records:
        for coord_field in _COORD_FIELDS:
            if coord_field in record:
                record[coord_field] = tuple(record[coord_field])
        identities.append(Identity(**record))
    return identities


def save_identities(identities: list[Identity], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for identity in identities:
        record = {field: getattr(identity, field) for field in _IDENTITY_FIELDS}
        for coord_field in _COORD_FIELDS:
            record[coord_field] = list(record[coord_field])
        records.append(record)
    p.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_instructions(path: str | Path) -> str:
    """The behavioral intro prepended to every agent's system prompt --
    kept as its own small JSON file (a list of instruction lines) rather
    than a string in code, so it's easy to tweak wording without touching
    Python.
    """
    p = Path(path)
    if not p.exists():
        return ""
    lines = json.loads(p.read_text(encoding="utf-8")).get("instructions", [])
    return "\n".join(f"- {line}" for line in lines)


def load_profile_template(path: str | Path) -> str:
    """The template an NPC's Identity gets rendered into (`{name}`, `{traits}`,
    etc.) -- lives in the same file as load_instructions() reads, so the
    whole static shape of what's sent to the LLM is visible in one place.
    Empty string if missing; the caller falls back to identity.DEFAULT_PROFILE_TEMPLATE.
    """
    p = Path(path)
    if not p.exists():
        return ""
    return json.loads(p.read_text(encoding="utf-8")).get("profile_template", "")


def load_memory(name: str, directory: str | Path) -> MemoryStore:
    """One file per NPC (`<name>.json`) rather than one shared file, so
    agents running concurrently later don't contend over the same file and
    a single NPC's memory can be inspected or reset independently.
    """
    store = MemoryStore()
    path = Path(directory) / f"{name}.json"
    if not path.exists():
        return store
    for record in json.loads(path.read_text(encoding="utf-8")):
        store.add(
            record["content"],
            importance=record["importance"],
            tags=set(record.get("tags", [])),
            timestamp=record["timestamp"],
        )
    return store


def save_memory(name: str, store: MemoryStore, directory: str | Path) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    records = [
        {"content": m.content, "importance": m.importance, "tags": sorted(m.tags), "timestamp": m.timestamp}
        for m in store.all()
    ]
    (d / f"{name}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_inventory(name: str, directory: str | Path) -> Inventory | None:
    """The NPC's inventory as of the last save, or None if there isn't one
    yet -- the caller then falls back to the identity's starting_money /
    starting_items. Same one-file-per-NPC layout as load_memory().
    """
    path = Path(directory) / f"{name}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    return Inventory(money=record["money"], items=record.get("items", {}))


def save_inventory(name: str, inventory: Inventory, directory: str | Path) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    record = {"money": inventory.money, "items": dict(sorted(inventory.items.items()))}
    (d / f"{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
