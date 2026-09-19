"""A network-free backend that behaves socially enough to watch -- what
GAME_AGENTS_LLM=mock runs in the mini-map (see bootstrap.py), so NPCs walk
their routines, seek each other out, and hold short canned conversations
without an API key.

It reads the same prompt a real model would get, via the formats this
package writes (world.render_surroundings(), the identity profile,
conversation.py's scene lines, a traveler's exit point), and picks one of
the offered tools. Randomness comes from one seeded Random, so a seeded run
is repeatable. Unit tests keep using llm.MockLLMClient, which never does
anything but echo.
"""
from __future__ import annotations

import random
import re
import threading
from typing import Any

from .llm import ENDS_CONVERSATION_FIELD, SPEAK_TOOL_NAME, LLMResult, MockLLMClient, ToolCall
from .world import BUSY_MARK, IN_REACH_LABEL, IN_VIEW_LABEL, INTERACTION_RANGE, clamp_coordinate, distance

_COORD = r"\((-?\d+), (-?\d+)\)"
_PERSON = re.compile(rf"(.+?) at {_COORD}({re.escape(BUSY_MARK)})?$")

_OPENERS = [
    "Hello there, {partner}! Fine {weather} day, isn't it?",
    "{partner}! Got a moment?",
    "Well met, {partner}. What brings you this way?",
    "Afternoon, {partner}. Anything new around town?",
]
_REPLIES = [
    "Can't complain. Busy, mostly.",
    "Oh, you know how it is. Same as always.",
    "Funny you ask -- I was just thinking about that.",
    "Is that so? Tell me more.",
    "Ha! Never would have guessed.",
    "Hard to say. The {weather} weather has everyone slow.",
    "I heard the same thing yesterday.",
    "Keep that between us, would you?",
]
_GOODBYES = [
    "Well, I'd best be off. Take care, {partner}.",
    "Good talking to you, {partner}. See you around.",
    "Right, work won't do itself. Bye now.",
    "Safe travels, {partner}.",
]
_MUTTERS = [
    "Hm. Quiet today.",
    "Where did I leave that...",
    "Nice {weather} weather, at least.",
    "Long day ahead.",
]


class ChattyMockLLMClient:
    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        # One client serves every NPC from several threads.
        self._lock = threading.Lock()

    def complete(
        self, *, system: str, messages: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> LLMResult:
        speak = next((t for t in tools if t["name"] == SPEAK_TOOL_NAME), None)
        tool_names = {t["name"] for t in tools}
        if speak is None and "wait" not in tool_names:
            # Not an NPC turn (e.g. traveler identity generation): answer
            # like the plain mock, which makes the caller use its fallback.
            # (An NPC turn can have speaking taken away, but never waiting.)
            return MockLLMClient().complete(system=system, messages=messages, tools=tools)
        stimulus = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        with self._lock:
            if speak is not None and ENDS_CONVERSATION_FIELD in speak["parameters"]["properties"]:
                return self._converse(system, stimulus, tool_names)
            if _coords(r"exit point: " + _COORD, system) is not None:
                return self._act_traveler(system, stimulus, tool_names)
            return self._act_resident(system, tool_names)

    # ------------------------------------------------------------------ #
    def _converse(self, system: str, stimulus: str, tools: set[str]) -> LLMResult:
        me = _search(r"^You are ([^,.\n]+)", system) or ""
        partner = _search(r"(?:chosen to talk to|in a conversation with) ([^,\n]+?)(?:, who|\.)", system) or "friend"
        fill = {"partner": partner, "weather": _search(r"and ([\w ]+?) out\.", system) or "fine"}
        said = len(re.findall(rf"^{re.escape(me)}: ", system, re.M)) if me else 0

        if "is wrapping up the conversation" in system:
            return _say(self._rng.choice(_GOODBYES), fill, ends=True)

        # A traveler here to buy something, talking to the one who sells it:
        # ask, then buy at whatever price was named.
        errand = _errand(system)
        if errand is not None and errand[1] == partner and not _bought(system, stimulus):
            item, seller = errand
            price = _search(r"(\d+) coins", stimulus)
            if price is not None and "buy_item" in tools:
                return _call("buy_item", item=item, seller_name=seller, total_price=int(price))
            return _say(f"Hello, {seller}. I'm after {_a(item)}. What's your price?", fill)
        # The other side of that: name a price.
        asked = re.search(r"I'm after (?:an? )?(.+?)\. What's your price\?", stimulus)
        if asked is not None:
            return _say(f"{self._rng.randint(3, 8)} coins for the {asked.group(1)}, and it's yours.", fill)

        if "Open the conversation" in system:
            return _say(self._rng.choice(_OPENERS), fill)
        if said >= 3 or (said >= 1 and self._rng.random() < 0.35):
            return _say(self._rng.choice(_GOODBYES), fill, ends=True)
        return _say(self._rng.choice(_REPLIES), fill)

    def _act_traveler(self, system: str, stimulus: str, tools: set[str]) -> LLMResult:
        """Plays whichever purpose the traveler's prompt gives it (see
        mini_map/traveler_purpose.py): a buyer goes straight for its seller,
        a visitor heads for its place and chats with whoever's there, and
        anyone passing through (or done) makes for the exit, only rarely
        stopping to talk.
        """
        rng = self._rng
        here = _coords(r"You are currently at " + _COORD, system) or (0, 0)
        walking = "You are walking toward" in system
        exit_point = _coords(r"exit point: " + _COORD, system)
        in_reach = [p for p in _people(IN_REACH_LABEL, system) if _fresh(p[0], system)]
        in_view = [p for p in _people(IN_VIEW_LABEL, system) if _fresh(p[0], system)]
        place = _coords(r"(?:spend some time at|who works at) .+? at " + _COORD, system)
        leave = _call("move_to", x=exit_point[0], y=exit_point[1])

        errand = _errand(system)
        if errand is not None and not _bought(system, stimulus):
            seller = errand[1]
            near = next((p for p in _people(IN_REACH_LABEL, system) if p[0] == seller), None)
            if near is not None:
                return _call("wait") if near[2] else _call("initiate_conversation", target_name=seller)
            seen = next((p for p in _people(IN_VIEW_LABEL, system) if p[0] == seller), None)
            if seen is not None:
                return _call("move_to", **_beside(seen[1], here))
            if place is not None and distance(here, place) > INTERACTION_RANGE and not walking:
                return _call("move_to", **_beside(place, here))
            return _call("wait")  # at the place; wait for the seller to turn up

        if "spend some time at" in system:
            arrived = "You're spending time at" in stimulus or (place is not None and distance(here, place) <= INTERACTION_RANGE)
            if not arrived:
                if in_reach and rng.random() < 0.3:
                    return _call("initiate_conversation", target_name=rng.choice(in_reach)[0])
                if walking:
                    return _call("wait")
                return _call("move_to", **_beside(place, here))
            free = [p for p in in_reach if not p[2]]
            if free:
                return _call("initiate_conversation", target_name=rng.choice(free)[0])
            free = [p for p in in_view if not p[2]]
            if free and rng.random() < 0.5:
                return _call("move_to", **_beside(rng.choice(free)[1], here))
            # Linger; only once it has stood around a while does it
            # sometimes decide it's had enough.
            if walking or "not walking anywhere" not in stimulus or rng.random() < 0.6:
                return _call("wait")
            return leave

        # Passing through, or done with what it came for.
        free = [p for p in in_reach if not p[2]]
        if free and rng.random() < 0.15:
            return _call("initiate_conversation", target_name=rng.choice(free)[0])
        return _call("wait") if walking else leave

    def _act_resident(self, system: str, tools: set[str]) -> LLMResult:
        rng = self._rng
        here = _coords(r"You are currently at " + _COORD, system) or (0, 0)
        free = [p for p in _people(IN_REACH_LABEL, system) if not p[2] and _fresh(p[0], system)]
        if free and rng.random() < 0.5:
            return _call("initiate_conversation", target_name=rng.choice(free)[0])
        if "You are walking toward" in system:
            return _call("wait")

        # Work by day, home by night.
        phase = _search(r"^Time: .*\((\w+)\)", system) or "morning"
        goal = _coords(("Home: " if phase in ("evening", "night") else "Workplace: ") + _COORD, system)
        if goal is not None and distance(here, goal) > 3 and "move_to" in tools:
            return _call("move_to", x=goal[0], y=goal[1])
        if SPEAK_TOOL_NAME in tools and rng.random() < 0.2:
            fill = {"weather": _search(r"and ([\w ]+?) out\.", system) or "fine"}
            return _say(rng.choice(_MUTTERS), fill)
        return _call("wait")


def _fresh(name: str, system: str) -> bool:
    """Not someone this NPC remembers already talking to."""
    return f"conversation with {name}" not in system and f"{name} says:" not in system


def _errand(system: str) -> tuple[str, str] | None:
    """(item, seller) for a traveler that came to buy something."""
    match = re.search(r"to buy (.+?) from (.+?), who works at", system)
    return (match.group(1), match.group(2)) if match else None


def _bought(system: str, stimulus: str) -> bool:
    """Done with the errand: bought it -- or tried and the trade fell
    through (sold out, too dear), which a mock can't haggle past."""
    return "You bought " in system or "You've bought " in stimulus or "The trade didn't go through" in system


def _a(item: str) -> str:
    return f"{'an' if item[:1] in 'aeiou' else 'a'} {item}"


def _search(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.M)
    return match.group(1).strip() if match else None


def _coords(pattern: str, text: str) -> tuple[int, int] | None:
    match = re.search(pattern, text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _people(label: str, system: str) -> list[tuple[str, tuple[int, int], bool]]:
    line = _search(rf"^{re.escape(label)}: (.+)\.$", system)
    if not line:
        return []
    people = []
    for entry in line.split("; "):
        match = _PERSON.match(entry)
        if match:
            people.append((match.group(1), (int(match.group(2)), int(match.group(3))), bool(match.group(4))))
    return people


def _beside(target: tuple[int, int], here: tuple[int, int], gap: float = 7.0) -> dict[str, int]:
    """A spot `gap` units short of `target` on the way from `here`, so the
    walker stops next to someone instead of on top of them -- still inside
    INTERACTION_RANGE, so they can talk once there.
    """
    dist = distance(here, target)
    if dist <= gap:
        return {"x": int(here[0]), "y": int(here[1])}
    t = (dist - gap) / dist
    return {
        "x": clamp_coordinate(round(here[0] + (target[0] - here[0]) * t)),
        "y": clamp_coordinate(round(here[1] + (target[1] - here[1]) * t)),
    }


def _say(template: str, fill: dict[str, str], *, ends: bool = False) -> LLMResult:
    arguments: dict[str, Any] = {"text": template.format(**{"partner": "", "weather": "fine", **fill})}
    if ends:
        arguments[ENDS_CONVERSATION_FIELD] = True
    return LLMResult(tool_call=ToolCall(name=SPEAK_TOOL_NAME, arguments=arguments))


def _call(name: str, **arguments: Any) -> LLMResult:
    return LLMResult(tool_call=ToolCall(name=name, arguments=arguments))
