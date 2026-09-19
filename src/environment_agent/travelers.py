from __future__ import annotations

import random
from dataclasses import dataclass, field

# Word pools for invent_traveler() below. The sketch they produce is only a
# brief: the full identity is generated from it by an LLM elsewhere (see
# game_agents/traveler_identity.py), with this sketch as the fallback.
# Small and easy to hand-edit; not meant to be exhaustive.
_FIRST_NAMES = [
    ("Odell", "male"),
    ("Brynn", "female"),
    ("Tamsin", "female"),
    ("Cael", "male"),
    ("Rosalind", "female"),
    ("Merrick", "male"),
    ("Ysolde", "female"),
    ("Doran", "male"),
    ("Anselm", "male"),
    ("Isolde", "female"),
    ("Corwin", "male"),
    ("Maren", "female"),
    ("Aldric", "male"),
    ("Linnea", "female"),
    ("Tobin", "male"),
    ("Sabine", "female"),
    ("Halvard", "male"),
    ("Wynne", "female"),
    ("Evander", "male"),
    ("Petra", "female"),
    ("Lucan", "male"),
    ("Odette", "female"),
    ("Garrick", "male"),
    ("Elspeth", "female"),
]
_ORIGINS = ["a mountain pass to the north", "the coastal trade roads", "a caravan out of the east", "no town in particular"]
_REASONS = [
    "passing through on the way to somewhere else",
    "looking to trade before moving on",
    "chasing down a rumor",
    "delivering a message",
    "just curious about the town",
]
_TRAITS = ["weary", "guarded", "talkative", "well-dressed", "travel-worn", "watchful"]


@dataclass
class Traveler:
    """A one-off visitor invented at spawn time -- not an NPC: no home,
    workplace, or persistent memory of its own (see the project's core
    invariants). If NPCs ever need to remember one, that memory lives on
    the NPC they spoke to, not here.
    """

    name: str
    origin: str
    reason: str
    traits: list[str] = field(default_factory=list)
    gender: str = ""


def spare_names(gender: str) -> list[str]:
    """Every pool name of that gender (all of them for an unknown one) --
    for renaming a traveler whose name turns out to be taken.
    """
    return [name for name, g in _FIRST_NAMES if g == gender] or [name for name, _ in _FIRST_NAMES]


def invent_traveler(rng: random.Random) -> Traveler:
    """Rolls a traveler sketch from the word pools above."""
    name, gender = rng.choice(_FIRST_NAMES)
    return Traveler(
        name=name,
        gender=gender,
        origin=rng.choice(_ORIGINS),
        reason=rng.choice(_REASONS),
        traits=rng.sample(_TRAITS, k=2),
    )
