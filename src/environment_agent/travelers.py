from __future__ import annotations

import random
from dataclasses import dataclass, field

# Word pools for invent_traveler() below -- a deterministic stand-in for what
# should eventually be an LLM call ("invent a traveler passing through
# today"). Small and easy to hand-edit for now; not meant to be exhaustive.
_FIRST_NAMES = ["Odell", "Brynn", "Tamsin", "Cael", "Rosalind", "Merrick", "Ysolde", "Doran"]
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


def invent_traveler(rng: random.Random) -> Traveler:
    """Rolls a traveler from the word pools above. Swap this body for an
    LLM call later; callers only ever see "give me a traveler" in, a
    Traveler out.
    """
    return Traveler(
        name=rng.choice(_FIRST_NAMES),
        origin=rng.choice(_ORIGINS),
        reason=rng.choice(_REASONS),
        traits=rng.sample(_TRAITS, k=2),
    )
