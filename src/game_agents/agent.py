from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .identity import DEFAULT_PROFILE_TEMPLATE, Identity
from .inventory import Inventory
from .llm import (
    ENDS_CONVERSATION_FIELD,
    PLAYER_NAME_FIELD,
    SPEAK_TOOL_NAME,
    SPEAK_TOOL_SCHEMA,
    LLMClient,
    LLMResult,
)
from .memory import Memory, MemoryStore
from .tools import ToolRegistry


@dataclass
class Scene:
    """Situational context for a single stimulus: when, where, why."""

    time: str = ""
    location: str = ""
    context: str = ""

    def prompt_block(self) -> str:
        lines = [
            f"{label}: {value}"
            for label, value in (("Time", self.time), ("Location", self.location), ("Context", self.context))
            if value
        ]
        return "\n".join(lines)


# Every action an NPC actually takes, one INFO record each -- mini_map.game
# prints these to the console.
action_log = logging.getLogger("game_agents.actions")

# Actions that don't change anything anyone else would notice -- on a
# routine turn (see Agent.respond), taking one of these isn't worth a memory.
ROUTINE_ACTIONS = frozenset({"wait", "move_to", "check_inventory"})

# Actions not offered on a conversation turn: you're already talking to
# someone -- offering to start a conversation there invites a model to
# "start" the one it's in (and so end it without saying a word).
CONVERSATION_HIDDEN_ACTIONS = frozenset({"initiate_conversation"})

# The opposite: actions only offered on a conversation turn. A trade needs
# both sides to agree to it, and the only place the other side gets a say
# is a conversation -- see also the registry's trade tools, which refuse
# outside one.
CONVERSATION_ONLY_ACTIONS = frozenset({"buy_item", "sell_item"})

# Only offered while talking with the player (see respond()'s
# `with_player`): noting their name, or selling to them, makes no sense
# with anyone else.
PLAYER_ONLY_ACTIONS = frozenset({"note_player_name", "sell_to_player"})

# What a model fills PLAYER_NAME_FIELD with when the player hasn't actually
# given a name -- taken as "no name yet", not as what to call them.
NON_NAMES = frozenset({"player", "the player", "unknown", "stranger", "none", "n/a", "traveler", "traveller"})


@dataclass
class TurnResult:
    utterance: str | None
    action: dict[str, Any] | None
    # Only ever True on a conversation turn (see respond()'s `conversation`):
    # the speaker marked this line as their goodbye.
    ends_conversation: bool = False


class Agent:
    """One NPC: a fixed identity, its own memory, and whatever actions it's
    been given, wired to an LLM. `respond()` is the entire loop: retrieve ->
    prompt -> call -> record -> return.
    """

    # How far past max_utterance_words a line may run before it's trimmed.
    HARD_CAP_FACTOR = 2

    def __init__(
        self,
        identity: Identity,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        memory: MemoryStore | None = None,
        instructions: str = "",
        profile_template: str = DEFAULT_PROFILE_TEMPLATE,
        position: tuple[float, float] | None = None,
        inventory: Inventory | None = None,
        standing_context: str = "",
        max_utterance_words: int | None = None,
    ) -> None:
        self.identity = identity
        self.memory = memory if memory is not None else MemoryStore()
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.instructions = instructions
        self.profile_template = profile_template
        # Extra, always-on lines for this one agent's system prompt, right
        # after its identity -- e.g. a traveler's exit point.
        self.standing_context = standing_context
        # None = unlimited. Otherwise the speak tool asks the model to keep
        # to it, and only a line well past it (HARD_CAP_FACTOR times) gets
        # cut -- back to its last whole sentence (see _cap_utterance()).
        self.max_utterance_words = max_utterance_words
        # Dynamic, unlike home/workplace on Identity -- spawns at home. The
        # move tool (see registry.py) only sets `destination`; the NPC then
        # actually walks there over time, one NPCRegistry.step_movement()
        # at a time, carrying `velocity` between steps.
        self.position = position if position is not None else identity.home
        self.velocity: tuple[float, float] = (0.0, 0.0)
        self.destination: tuple[int, int] | None = None
        # Same idea: seeded from the identity's starting money/items, then
        # changed only by trades (see registry.py's buy/sell tools).
        self.inventory = (
            inventory
            if inventory is not None
            else Inventory(money=identity.starting_money, items=dict(identity.starting_items))
        )
        # The player's name, once they've said it (see the registry's
        # note_player_name tool) -- until then they're a stranger.
        self.player_name: str | None = None

    def player_label(self) -> str:
        """How this NPC refers to the player: by name once they know it."""
        if self.player_name:
            return f"{self.player_name} (the player)"
        return "the player (you don't know their name yet)"

    def respond(
        self,
        stimulus: str,
        *,
        scene: Scene | None = None,
        tags: set[str] | None = None,
        routine: bool = False,
        conversation: bool = False,
        with_player: bool = False,
        hide: frozenset[str] = frozenset(),
    ) -> TurnResult:
        """`routine` marks a turn nobody prompted (e.g. the world tick's
        "It is now 08:15"): it's only remembered if the NPC did something
        consequential (a trade, starting a conversation). Otherwise every
        tick's "walked to (x, y)" would crowd real interactions out of the
        top memories -- current position and destination are already in the
        system prompt, so nothing is lost by dropping them.

        `conversation` marks a turn inside an NPC-to-NPC exchange: the speak
        tool then also offers `ends_conversation`, so a goodbye can close
        the exchange on the same line instead of costing an extra turn.
        Trades (CONVERSATION_ONLY_ACTIONS) are only offered on one.

        `with_player` marks a turn talking with the player, the only place
        PLAYER_ONLY_ACTIONS are offered.

        `hide` takes options off the table for this one turn, by tool name
        -- SPEAK_TOOL_NAME included, e.g. so an NPC that just said goodbye
        has to actually leave instead of saying it again.
        """
        scene = scene or Scene()
        relevant = self.memory.retrieve(tags=tags)
        system = self._build_system_prompt(scene, relevant)

        def offer(hide: frozenset[str]) -> list[dict[str, Any]]:
            return [
                *(
                    []
                    if SPEAK_TOOL_NAME in hide
                    else [self._speak_schema(conversation=conversation, with_player=with_player)]
                ),
                *(
                    schema
                    for schema in self.tools.schemas()
                    if not self._hidden(schema["name"], conversation, hide, with_player=with_player)
                ),
            ]

        offered = offer(hide)
        result = self.llm.complete(system=system, messages=[{"role": "user", "content": stimulus}], tools=offered)
        call = result.tool_call
        allowed = call.name in {schema["name"] for schema in offered}

        # Only a look, not a move (look_around, check_inventory): hand what
        # they found straight back and let them decide again, this same
        # turn -- otherwise looking would cost the whole turn (and, talking
        # to the player, leave them with no reply). Each look is offered
        # once per turn, so this always ends. Not stored: where everyone
        # stood a moment ago goes stale fast.
        looked: set[str] = set()
        notes: list[str] = []
        while allowed and (tool := self.tools.get(call.name)) is not None and tool.informational:
            found = self.tools.execute(call.name, call.arguments)
            action_log.info("%s: %s() -> %s", self.identity.name, call.name, found)
            looked.add(call.name)
            notes.append(f"(You {call.name.replace('_', ' ')}: {found})")
            offered = offer(hide | looked)
            result = self.llm.complete(
                system=system,
                messages=[{"role": "user", "content": "\n\n".join([stimulus, *notes])}],
                tools=offered,
            )
            call = result.tool_call
            allowed = call.name in {schema["name"] for schema in offered}

        ends_conversation = False
        if not allowed:
            # The real backends can only pick an offered tool; this is for
            # any that doesn't -- a hidden action stays undone, and isn't
            # worth remembering.
            tool_result = "That isn't something you can do right now."
            return TurnResult(utterance=None, action={"name": call.name, "arguments": call.arguments, "result": tool_result})
        if call.name == SPEAK_TOOL_NAME:
            utterance = self._cap_utterance(call.arguments.get("text", ""))
            action = None
            ends_conversation = conversation and bool(call.arguments.get(ENDS_CONVERSATION_FIELD, False))
            if with_player:
                self._note_player_name(call.arguments.get(PLAYER_NAME_FIELD))
            memory_content = f"{stimulus} -> {utterance}"
        else:
            utterance = None
            tool_result = self.tools.execute(call.name, call.arguments)
            args = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
            action_log.info("%s: %s(%s) -> %s", self.identity.name, call.name, args, tool_result)
            action = {"name": call.name, "arguments": call.arguments, "result": tool_result}
            memory_content = f"{stimulus} -> [action] {call.name}({call.arguments}) -> {tool_result}"

        turn = TurnResult(utterance=utterance, action=action, ends_conversation=ends_conversation)
        if routine and (action is None or action["name"] in ROUTINE_ACTIONS):
            return turn

        self.memory.add(memory_content, importance=self._score_importance(result), tags=tags or set())

        return turn

    @staticmethod
    def _hidden(name: str, conversation: bool, hide: frozenset[str], *, with_player: bool = False) -> bool:
        if name in hide or (name in PLAYER_ONLY_ACTIONS and not with_player):
            return True
        return name in (CONVERSATION_HIDDEN_ACTIONS if conversation else CONVERSATION_ONLY_ACTIONS)

    def _note_player_name(self, given: Any) -> None:
        """The name the player gave, caught on the same line that heard it
        (see PLAYER_NAME_FIELD). Only ever set once: whatever they said
        first is their name, and a later line mentioning someone else
        shouldn't rename them.
        """
        if self.player_name is not None or not isinstance(given, str):
            return
        cleaned = " ".join(given.split())
        # Models fill the field with their own name, or a placeholder, when
        # the player hasn't actually said one.
        if not cleaned or cleaned.lower() in NON_NAMES or cleaned.lower() == self.identity.name.lower():
            return
        self.player_name = cleaned

    def _speak_schema(self, *, conversation: bool = False, with_player: bool = False) -> dict[str, Any]:
        schema = SPEAK_TOOL_SCHEMA
        # Outside a conversation, a spoken line has no one to answer it --
        # models kept using it to address people instead of talking with
        # them, so it's described as what it is.
        description = (
            schema["description"]
            if conversation
            else "Think aloud, in character: a remark to yourself that anyone nearby might overhear. "
            "It is not a conversation -- to talk with someone, start a conversation with them instead."
        )
        if self.max_utterance_words is not None:
            description += f" You're a person of few words: at most {self.max_utterance_words} words."
        if description != schema["description"]:
            schema = {**schema, "description": description}
        # Extra fields, folded in together so a turn can carry both -- a
        # goodbye that also catches a name shouldn't have to pick one.
        extra: dict[str, Any] = {}
        if conversation:
            extra[ENDS_CONVERSATION_FIELD] = {
                "type": "boolean",
                "description": (
                    "True if this line is your goodbye and the conversation is over. "
                    "Leave it false while there's still something to say."
                ),
            }
        if with_player and self.player_name is None:
            extra[PLAYER_NAME_FIELD] = {
                "type": "string",
                "description": (
                    "The player's own name, if they've told you it by now -- theirs, not yours. "
                    "Fill it in on the line you hear it and you'll know them by it from then on. "
                    "Leave it out until they've actually said it."
                ),
            }
        if extra:
            parameters = schema["parameters"]
            schema = {
                **schema,
                "parameters": {**parameters, "properties": {**parameters["properties"], **extra}},
            }
        return schema

    def _cap_utterance(self, text: str) -> str:
        """The word limit is a request, not a guillotine: cutting a line at
        exactly N words left half-sentences the other side then answered.
        Only a line well past it is trimmed, to its last complete sentence
        within the hard cap (or cut with "..." if it has none).
        """
        if self.max_utterance_words is None:
            return text
        hard_cap = self.max_utterance_words * self.HARD_CAP_FACTOR
        words = text.split()
        if len(words) <= hard_cap:
            return text
        kept = " ".join(words[:hard_cap])
        end = max(kept.rfind(mark) for mark in ".!?")
        if end > 0:
            return kept[: end + 1]
        return kept + "..."

    def _build_system_prompt(self, scene: Scene, memories: list[Memory]) -> str:
        parts = []
        if self.instructions:
            parts.append(self.instructions)
        parts.append(self.identity.prompt_block(self.profile_template))
        if self.standing_context:
            parts.append(self.standing_context)
        parts.append(f"You are currently at ({self.position[0]:.0f}, {self.position[1]:.0f}).")
        if self.destination is not None:
            parts.append(f"You are walking toward ({self.destination[0]}, {self.destination[1]}).")
        scene_block = scene.prompt_block()
        if scene_block:
            parts.append(scene_block)
        if memories:
            parts.append("Relevant memories:\n" + "\n".join(f"- {m.content}" for m in memories))
        return "\n\n".join(parts)

    def _score_importance(self, result: LLMResult) -> int:
        # Heuristic for v1: a turn that took an action outranks plain small
        # talk. Swap for an LLM-rated score later if this proves too coarse.
        return 7 if result.tool_call.name != SPEAK_TOOL_NAME else 4
