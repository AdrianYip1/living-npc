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

# Every Identity field except home/workplace: a traveler lives nowhere in
# town, so those stay at whatever the fallback identity carries.
CREATE_IDENTITY_TOOL_SCHEMA: dict[str, Any] = {
    "name": CREATE_IDENTITY_TOOL_NAME,
    "description": "Create the full identity of a traveler arriving in town.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "First name only, one word."},
            "traits": {"type": "array", "items": {"type": "string"}, "description": "2-4 short personality traits."},
            "backstory": {"type": "string", "description": "One or two sentences: who they are and why they're here."},
            "speech_style": {"type": "string", "description": "How they talk, in a few words."},
            "goals": {"type": "array", "items": {"type": "string"}, "description": "1-2 things they want during this visit."},
            "habits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-3 short lines on how they spend their brief time in town.",
            },
            "starting_money": {"type": "integer", "minimum": 0, "description": "Coins they carry."},
            "starting_items": {
                "type": "object",
                "additionalProperties": {"type": "integer", "minimum": 1},
                "description": "Item name -> count; what a traveler like this would carry.",
            },
        },
        "required": [
            "name",
            "traits",
            "backstory",
            "speech_style",
            "goals",
            "habits",
            "starting_money",
            "starting_items",
        ],
    },
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

    if result.tool_call.name != CREATE_IDENTITY_TOOL_NAME:
        return fallback

    identity = _parse(result.tool_call.arguments, fallback)
    if identity is None or identity.name.lower() in taken:
        return fallback
    return identity


def _parse(arguments: dict[str, Any], fallback: Identity) -> Identity | None:
    fields = CREATE_IDENTITY_TOOL_SCHEMA["parameters"]["properties"]
    record = {key: arguments[key] for key in fields if key in arguments}
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
    return identity
