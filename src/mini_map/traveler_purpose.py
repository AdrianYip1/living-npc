"""Why a traveler stopped in town -- rolled once on arrival (see
Simulation._admit_traveler) and threaded through everything the traveler is
told: its generated identity, its standing prompt, and the reminder tacked
onto each of its turns.

Three kinds, picked at random:
- passing through: no stop planned, straight to the exit point.
- socialize: spend some time at one of world.json's places, chatting with
  whoever's there, then leave.
- buy: get one specific item from the resident who works at one of those
  places -- an item that resident actually has, so the trade can happen.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from game_agents.agent import Agent

PASSING_THROUGH = "passing_through"
SOCIALIZE = "socialize"
BUY = "buy"
KINDS = (PASSING_THROUGH, SOCIALIZE, BUY)


@dataclass
class TravelerPurpose:
    kind: str
    # The world.json place it's headed to (socialize, buy); None when just
    # passing through.
    place: dict[str, Any] | None = None
    # buy only: what, and from whom.
    item: str | None = None
    seller: str | None = None

    @property
    def place_label(self) -> str:
        if self.place is None:
            return ""
        return f"{self.place['name']} at ({self.place['position'][0]}, {self.place['position'][1]})"

    def brief(self) -> str:
        """One sentence for the identity generator's brief, so the invented
        person fits what they're here to do.
        """
        if self.kind == SOCIALIZE:
            return f"They've stopped in town to spend a while at {self.place['name']}, resting and catching up on news."
        if self.kind == BUY:
            return f"They've stopped in town to buy {self.item} from {self.seller} at {self.place['name']}."
        return "They're only passing through on the way somewhere else."

    def summary(self) -> str:
        """Short, third-person, for the page: "here to buy horseshoe from Mara"."""
        if self.kind == SOCIALIZE:
            return f"here to spend some time at {self.place['name']}"
        if self.kind == BUY:
            return f"here to buy {self.item} from {self.seller} at {self.place['name']}"
        return "just passing through"

    def goal(self) -> str:
        """The same thing as a goal line, for an identity with no LLM behind it."""
        if self.kind == SOCIALIZE:
            return f"spend some time at {self.place['name']}"
        if self.kind == BUY:
            return f"buy {self.item} from {self.seller}"
        return "get where they're going"

    def standing_context(self) -> str:
        """Always-on lines for the traveler's system prompt."""
        if self.kind == SOCIALIZE:
            return (
                f"You've stopped in town to spend some time at {self.place_label}: rest, catch up on news, "
                "and chat with whoever's around -- an hour or two, not all day. Then head for your exit point."
            )
        if self.kind == BUY:
            return (
                f"You've stopped in town to buy {self.item} from {self.seller}, who works at {self.place_label}. "
                f"Find {self.seller}, agree on a price out loud, then use buy_item. Once you have it, head for your exit point."
            )
        return (
            "You're just passing through on your way somewhere else -- there's nowhere in town you need to stop, "
            "though you can if something catches your interest."
        )

    def reminder(self, *, arrived: bool, bought: bool) -> str:
        """Tacked onto a turn's stimulus: where the traveler stands on its
        purpose right now. `arrived` means it has been to the place;
        `bought` that it has the item.
        """
        if self.kind == SOCIALIZE:
            if arrived:
                return f" You're spending time at {self.place['name']} -- an hour or two is plenty, then head on."
            return f" You still want to spend some time at {self.place_label}."
        if self.kind == BUY:
            if bought:
                return f" You've bought the {self.item} you came for."
            return f" You still want to buy {self.item} from {self.seller} at {self.place_label}."
        return " You're just passing through."


def pick_purpose(rng: random.Random, places: list[dict[str, Any]], residents: list[Agent]) -> TravelerPurpose:
    """Rolls a purpose, falling back to one that's actually possible: no
    places means passing through, and no resident with anything to sell at
    the chosen place turns a buy into a visit.
    """
    kind = rng.choice(KINDS)
    if kind == PASSING_THROUGH or not places:
        return TravelerPurpose(PASSING_THROUGH)
    place = rng.choice(places)
    if kind == BUY:
        sellers = [agent for agent in residents if list(agent.identity.workplace) == list(place["position"])]
        seller = rng.choice(sellers) if sellers else None
        if seller is not None and seller.inventory.items:
            # Something they have more than one of, if they can -- a
            # blacksmith's spare horseshoes, not the hammer she works with.
            spare = [item for item, count in seller.inventory.items.items() if count > 1]
            item = rng.choice(sorted(spare or seller.inventory.items))
            return TravelerPurpose(BUY, place=place, item=item, seller=seller.identity.name)
    return TravelerPurpose(SOCIALIZE, place=place)
