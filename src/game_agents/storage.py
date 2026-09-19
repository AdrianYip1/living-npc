from __future__ import annotations

import json
from pathlib import Path

from .identity import Identity
from .inventory import Inventory
from .memory import MemoryStore

_IDENTITY_FIELDS = ("name", "traits", "backstory", "speech_style", "goals", "home", "workplace", "habits", "starting_money", "starting_items")
_COORD_FIELDS = ("home", "workplace")


def identity_from_record(record: dict) -> Identity:
    """One npcs.json entry -> Identity. Unknown keys raise (via Identity's
    constructor), same as a typo in npcs.json would.
    """
    record = dict(record)
    for coord_field in _COORD_FIELDS:
        if coord_field in record:
            record[coord_field] = tuple(record[coord_field])
    return Identity(**record)


def identity_to_record(identity: Identity) -> dict:
    """Identity -> one npcs.json entry, JSON-ready."""
    record = {field: getattr(identity, field) for field in _IDENTITY_FIELDS}
    for coord_field in _COORD_FIELDS:
        record[coord_field] = list(record[coord_field])
    return record


def load_identities(path: str | Path) -> list[Identity]:
    p = Path(path)
    if not p.exists():
        return []
    return [identity_from_record(record) for record in json.loads(p.read_text(encoding="utf-8"))]


def save_identities(identities: list[Identity], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    records = [identity_to_record(identity) for identity in identities]
    p.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_instructions(path: str | Path, key: str = "instructions") -> str:
    """The behavioral intro prepended to every agent's system prompt --
    kept as its own small JSON file (a list of instruction lines) rather
    than a string in code, so it's easy to tweak wording without touching
    Python.
    """
    p = Path(path)
    if not p.exists():
        return ""
    lines = json.loads(p.read_text(encoding="utf-8")).get(key, [])
    return "\n".join(f"- {line}" for line in lines)


def load_profile_template(path: str | Path, key: str = "profile_template") -> str:
    """The template an NPC's Identity gets rendered into (`{name}`, `{traits}`,
    etc.) -- lives in the same file as load_instructions() reads, so the
    whole static shape of what's sent to the LLM is visible in one place.
    Empty string if missing; the caller falls back to identity.DEFAULT_PROFILE_TEMPLATE.
    """
    p = Path(path)
    if not p.exists():
        return ""
    return json.loads(p.read_text(encoding="utf-8")).get(key, "")


def load_places(path: str | Path) -> list[dict]:
    """world.json's named places -- each {"name", "position": [x, y],
    "description"}. Empty list if missing.
    """
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("places", [])


def render_places(places: list[dict]) -> str:
    """The prompt block every NPC (resident or traveler) sees, so anyone can
    move_to a place by its coordinates without having to already know them.
    """
    if not places:
        return ""
    lines = [
        f"- {place['name']} at ({place['position'][0]}, {place['position'][1]}): {place['description']}"
        for place in places
    ]
    return "Places in town you can walk to:\n" + "\n".join(lines)


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
