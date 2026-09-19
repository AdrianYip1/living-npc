"""Turns a short brief about an arriving traveler into a full Identity --
the same shape as a persistent NPC's npcs.json entry -- via the LLM.

Kept in game_agents (not environment_agent) because Identity is an NPC
concept: the environment agent only decides that someone arrives and
sketches who, and whoever ties the two together (mini_map.Simulation)
writes the brief and the fallback.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from .identity import Identity
from .llm import LLMClient
from .storage import identity_from_record

log = logging.getLogger(__name__)

CREATE_IDENTITY_TOOL_NAME = "create_identity"

# Every Identity field except home/workplace -- a traveler lives nowhere in
# town, so those stay at whatever the fallback identity carries -- and
# relationships / unfamiliar_with: they know no one here, and what's
# outside their field is left to the model.
CREATE_IDENTITY_TOOL_SCHEMA: dict[str, Any] = {
    "name": CREATE_IDENTITY_TOOL_NAME,
    "description": "Create the full identity of a traveler arriving in town.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "First name only, one word."},
            "gender": {"type": "string", "enum": ["male", "female"]},
            "age": {"type": "integer", "description": "Age in years."},
            "appearance": {"type": "string", "description": "What someone sees at a glance, in one short sentence."},
            "traits": {"type": "array", "items": {"type": "string"}, "description": "2-4 short personality traits."},
            "likes": {"type": "array", "items": {"type": "string"}, "description": "1-2 things they enjoy."},
            "dislikes": {"type": "array", "items": {"type": "string"}, "description": "1-2 things they can't stand."},
            "backstory": {"type": "string", "description": "One or two sentences: who they are and why they're here."},
            "speech_style": {"type": "string", "description": "How they talk, in a few words."},
            "goals": {"type": "array", "items": {"type": "string"}, "description": "1-2 things they want during this visit."},
            "habits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-3 short lines on how they spend their brief time in town.",
            },
            "starting_money": {"type": "integer", "description": "Coins they carry, 0 or more."},
            # A list of pairs, not an item -> count map: strict tool use
            # (below) can't express a map with free-form keys. _parse turns
            # it back into the dict Identity keeps.
            "starting_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "count": {"type": "integer", "description": "1 or more."},
                    },
                    "required": ["item", "count"],
                    "additionalProperties": False,
                },
                "description": "What a traveler like this would carry.",
            },
        },
        "required": [
            "name",
            "gender",
            "age",
            "appearance",
            "traits",
            "likes",
            "dislikes",
            "backstory",
            "speech_style",
            "goals",
            "habits",
            "starting_money",
            "starting_items",
        ],
        "additionalProperties": False,
    },
    # Without it the model regularly answered with a list field as one
    # comma-joined string or left a field out, and _parse rejected the whole
    # identity -- most travelers ended up on the fallback.
    "strict": True,
}

SYSTEM_PROMPT = (
    "You invent travelers passing through a small town, for a game. "
    "Given a short brief, create one believable, specific traveler by calling "
    f"{CREATE_IDENTITY_TOOL_NAME}. Keep every field short and grounded in the brief."
)


def generate_traveler_identity(
    llm: LLMClient,
    brief: str,
    *,
    fallback: Identity,
    taken_names: Iterable[str] = (),
) -> Identity:
    """Asks the LLM for a traveler's identity. Returns `fallback` instead if
    the call fails, the model answers with the wrong tool (e.g. the mock
    backend, which always speaks), the fields are malformed, or the name
    collides with one in taken_names -- an arrival never gets dropped
    because generation went wrong.
    """
    taken = {name.lower() for name in taken_names}
    content = brief
    if taken:
        content += f"\nDon't use any of these names: {', '.join(sorted(taken_names))}."

    try:
        result = llm.complete(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            tools=[CREATE_IDENTITY_TOOL_SCHEMA],
        )
    except Exception:
        log.exception("traveler identity generation failed; using fallback")
        return fallback

    # Every fallback is logged: a silent one looks just like a working
    # backend with dull travelers.
    if result.tool_call.name != CREATE_IDENTITY_TOOL_NAME:
        log.warning("traveler identity: model called %r instead; using fallback", result.tool_call.name)
        return fallback

    identity = _parse(result.tool_call.arguments, fallback)
    if identity is None:
        log.warning("traveler identity: malformed arguments %r; using fallback", result.tool_call.arguments)
        return fallback
    if identity.name.lower() in taken:
        log.warning("traveler identity: name %r is taken; using fallback", identity.name)
        return fallback
    return identity


def _parse(arguments: dict[str, Any], fallback: Identity) -> Identity | None:
    fields = CREATE_IDENTITY_TOOL_SCHEMA["parameters"]["properties"]
    record = {key: arguments[key] for key in fields if key in arguments}
    # Backends without strict tool use (DeepSeek) still sometimes send a
    # list as one "a, b, c" string -- close enough to take.
    for key in ("traits", "goals", "habits", "likes", "dislikes"):
        if isinstance(record.get(key), str):
            record[key] = [part.strip() for part in record[key].replace(";", ",").split(",") if part.strip()]
    items = record.get("starting_items")
    if isinstance(items, list):
        try:
            record["starting_items"] = {entry["item"]: entry["count"] for entry in items}
        except (TypeError, KeyError):
            return None
    record["home"] = fallback.home
    record["workplace"] = fallback.workplace
    try:
        identity = identity_from_record(record)
    except TypeError:  # missing required field
        return None

    def is_str_list(value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(item, str) for item in value)

    valid = (
        isinstance(identity.name, str)
        and identity.name.strip() != ""
        and is_str_list(identity.traits)
        and isinstance(identity.backstory, str)
        and isinstance(identity.speech_style, str)
        and is_str_list(identity.goals)
        and is_str_list(identity.habits)
        and isinstance(identity.starting_money, int)
        and identity.starting_money >= 0
        and isinstance(identity.starting_items, dict)
        and all(isinstance(k, str) and isinstance(v, int) and v > 0 for k, v in identity.starting_items.items())
    )
    if not valid:
        return None
    identity.name = identity.name.strip()
    # Only the speech side reads it, so a bad value isn't worth losing the
    # whole identity over -- the sketch's gender stands in.
    gender = identity.gender.strip().lower() if isinstance(identity.gender, str) else ""
    identity.gender = gender if gender in ("male", "female") else fallback.gender
    # Flavor, same as gender: a bad value is just left unstated.
    if not (isinstance(identity.age, int) and 0 < identity.age < 120):
        identity.age = 0
    if not isinstance(identity.appearance, str):
        identity.appearance = ""
    for key in ("likes", "dislikes"):
        if not is_str_list(getattr(identity, key)):
            setattr(identity, key, [])
    return identity
