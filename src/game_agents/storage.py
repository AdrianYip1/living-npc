from __future__ import annotations

import json
from pathlib import Path

from .identity import Identity
from .inventory import Inventory
from .memory import MemoryStore

_IDENTITY_FIELDS = ("name", "traits", "backstory", "speech_style", "goals", "home", "workplace", "habits", "starting_money", "starting_items", "gender", "age", "appearance", "likes", "dislikes", "unfamiliar_with", "relationships")
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
    "description"}, plus optional "spots": [[x, y], ...], other coordinates
    that are also inside the place (e.g. each resident's own corner of a
    shared house). Empty list if missing.
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


def load_town(path: str | Path) -> dict:
    """world.json's town -- {"name", "about": [line, ...]}: what the
    settlement is, that the listed places and people are all of it, and how
    far off everything else is. Empty dict if missing.
    """
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("town", {})


def render_town(town: dict) -> str:
    """The prompt block every NPC (resident or traveler) sees ahead of the
    places -- so the gaps between the listed places and people read as
    empty, not as room to invent a market or a mayor.
    """
    about = town.get("about", [])
    if not about:
        return ""
    return f"About {town.get('name', 'the town')}:\n" + "\n".join(f"- {line}" for line in about)


def load_prices(path: str | Path) -> dict[str, int]:
    """world.json's usual prices -- item name -> coins apiece. Empty dict
    if missing.
    """
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("prices", {})


def render_prices(prices: dict[str, int]) -> str:
    """What things usually cost -- a reference to haggle from, not a rule:
    buy_item / sell_item / sell_to_player take whatever total was agreed.
    """
    if not prices:
        return ""
    lines = [f"- {item}: {coins} coin{'s' if coins != 1 else ''}" for item, coins in prices.items()]
    return (
        "What things usually cost in town, per item (the only currency is coins):\n"
        + "\n".join(lines)
        + "\nA deal at a different price can happen, but not easily: only after real haggling or for a good "
        "reason. Don't drop far below or ask far above these without one. For anything not listed, judge "
        "by what's here."
    )


def render_residents(identities: list, places: list[dict]) -> str:
    """Where each resident can usually be found -- general town knowledge,
    not a live view: they move around, and look_around is how an NPC finds
    out where someone actually is right now.
    """
    if not identities:
        return ""
    names = {
        tuple(position): place["name"]
        for place in places
        for position in (place["position"], *place.get("spots", []))
    }

    def spot(position) -> str:
        position = tuple(position)
        name = names.get(position)
        return f"{name} ({position[0]}, {position[1]})" if name else f"({position[0]}, {position[1]})"

    lines = [f"- {i.name}: works at {spot(i.workplace)}; lives at {spot(i.home)}." for i in identities]
    return (
        "Where the townsfolk can usually be found (they move around, so they won't always be there):\n"
        + "\n".join(lines)
    )


def load_player_names(path: str | Path) -> dict[str, str]:
    """NPC name -> what that NPC knows the player as."""
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_player_names(names: dict[str, str], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(names, indent=2), encoding="utf-8")


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
