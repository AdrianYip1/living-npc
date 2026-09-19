from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .identity import DEFAULT_PROFILE_TEMPLATE, Identity
from .inventory import Inventory
from .llm import SPEAK_TOOL_NAME, SPEAK_TOOL_SCHEMA, LLMClient, LLMResult
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


# Actions that don't change anything anyone else would notice -- on a
# routine turn (see Agent.respond), taking one of these isn't worth a memory.
ROUTINE_ACTIONS = frozenset({"wait", "move_to", "check_inventory"})


@dataclass
class TurnResult:
    utterance: str | None
    action: dict[str, Any] | None


class Agent:
    """One NPC: a fixed identity, its own memory, and whatever actions it's
    been given, wired to an LLM. `respond()` is the entire loop: retrieve ->
    prompt -> call -> record -> return.
    """

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
    ) -> None:
        self.identity = identity
        self.memory = memory if memory is not None else MemoryStore()
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.instructions = instructions
        self.profile_template = profile_template
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

    def respond(
        self, stimulus: str, *, scene: Scene | None = None, tags: set[str] | None = None, routine: bool = False
    ) -> TurnResult:
        """`routine` marks a turn nobody prompted (e.g. the world tick's
        "It is now 08:15"): it's only remembered if the NPC did something
        consequential (a trade, starting a conversation). Otherwise every
        tick's "walked to (x, y)" would crowd real interactions out of the
        top memories -- current position and destination are already in the
        system prompt, so nothing is lost by dropping them.
        """
        scene = scene or Scene()
        relevant = self.memory.retrieve(tags=tags)

        result = self.llm.complete(
            system=self._build_system_prompt(scene, relevant),
            messages=[{"role": "user", "content": stimulus}],
            tools=[SPEAK_TOOL_SCHEMA, *self.tools.schemas()],
        )
        call = result.tool_call

        if call.name == SPEAK_TOOL_NAME:
            utterance = call.arguments.get("text", "")
            action = None
            memory_content = f"{stimulus} -> {utterance}"
        else:
            utterance = None
            tool_result = self.tools.execute(call.name, call.arguments)
            action = {"name": call.name, "arguments": call.arguments, "result": tool_result}
            memory_content = f"{stimulus} -> [action] {call.name}({call.arguments}) -> {tool_result}"

        if routine and (action is None or action["name"] in ROUTINE_ACTIONS):
            return TurnResult(utterance=utterance, action=action)

        self.memory.add(memory_content, importance=self._score_importance(result), tags=tags or set())

        return TurnResult(utterance=utterance, action=action)

    def _build_system_prompt(self, scene: Scene, memories: list[Memory]) -> str:
        parts = []
        if self.instructions:
            parts.append(self.instructions)
        parts.append(self.identity.prompt_block(self.profile_template))
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
