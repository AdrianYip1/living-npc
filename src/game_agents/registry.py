from __future__ import annotations

import time
from pathlib import Path

from .agent import Agent
from .conversation import run_conversation, save_transcript
from .identity import DEFAULT_PROFILE_TEMPLATE
from .llm import LLMClient
from .storage import load_identities, load_instructions, load_memory, load_profile_template, save_memory
from .tools import Tool, ToolRegistry


class NPCRegistry:
    """Loads every NPC in npcs.json into a live Agent, keyed by name (see
    Identity -- name doubles as the id), each with its own memory loaded
    from disk. The one place that knows how to find an NPC by name --
    anything that needs to address more than one NPC (a world loop,
    NPC-to-NPC visits, the player picking who to talk to) goes through this
    instead of constructing Agents itself.

    Also the one place that tracks which NPCs are currently busy (in a
    conversation), since that needs a shared source of truth reachable from
    every NPC's own initiate_conversation tool -- an Agent has no visibility
    into any other Agent on its own.
    """

    def __init__(
        self,
        npcs_path: str | Path,
        memory_dir: str | Path,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        instructions_path: str | Path | None = None,
        conversation_turns: int = 4,
        conversation_log_dir: str | Path | None = None,
    ) -> None:
        self._memory_dir = Path(memory_dir)
        self._conversation_log_dir = Path(conversation_log_dir) if conversation_log_dir else None
        self._busy: set[str] = set()
        instructions = load_instructions(instructions_path) if instructions_path else ""
        profile_template = (load_profile_template(instructions_path) if instructions_path else "") or DEFAULT_PROFILE_TEMPLATE
        common_tools = tools.all_tools() if tools is not None else []

        self._agents: dict[str, Agent] = {}
        for identity in load_identities(npcs_path):
            memory = load_memory(identity.name, self._memory_dir)
            npc_tools = ToolRegistry()
            for tool in common_tools:
                npc_tools.register(tool)
            npc_tools.register(self._make_initiate_conversation_tool(identity.name, conversation_turns))
            self._agents[identity.name] = Agent(
                identity, llm, npc_tools, memory=memory, instructions=instructions, profile_template=profile_template
            )

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    def all(self) -> list[Agent]:
        return list(self._agents.values())

    def save_all(self) -> None:
        for agent in self._agents.values():
            save_memory(agent.identity.name, agent.memory, self._memory_dir)

    # ------------------------------------------------------------------ #
    # busy tracking
    # ------------------------------------------------------------------ #
    def is_busy(self, name: str) -> bool:
        return name in self._busy

    def try_occupy_pair(self, a_name: str, b_name: str) -> bool:
        """Claims both names at once, or neither. Also what stops a
        conversation from re-initiating mid-exchange: a side already busy
        (both participants are marked busy for the whole exchange, including
        the initiator) can never successfully claim a new pair, so a nested
        initiate_conversation call just fails harmlessly instead of
        recursing.
        """
        if self.is_busy(a_name) or self.is_busy(b_name):
            return False
        self._busy.add(a_name)
        self._busy.add(b_name)
        return True

    def release_pair(self, a_name: str, b_name: str) -> None:
        self._busy.discard(a_name)
        self._busy.discard(b_name)

    # ------------------------------------------------------------------ #
    def _make_initiate_conversation_tool(self, initiator_name: str, turns: int) -> Tool:
        def handler(target_name: str) -> str:
            target = self._resolve(target_name)
            if target is None:
                return f"There's no one named {target_name!r} nearby."
            resolved_name = target.identity.name
            if not self.try_occupy_pair(initiator_name, resolved_name):
                return f"{resolved_name} is busy right now."
            try:
                transcript = run_conversation(self.get(initiator_name), target, turns=turns)
                if self._conversation_log_dir is not None:
                    log_path = (
                        self._conversation_log_dir / f"{initiator_name}_{resolved_name}_{int(time.time() * 1000)}.json"
                    )
                    save_transcript(transcript, log_path)
            finally:
                self.release_pair(initiator_name, resolved_name)
            return f"You had a conversation with {resolved_name}."

        return Tool(
            name="initiate_conversation",
            description=(
                "Walk up to another NPC who's nearby and start a conversation with "
                "them. Everyone is assumed to be nearby for now."
            ),
            parameters={
                "type": "object",
                "properties": {"target_name": {"type": "string", "description": "The name of the NPC to talk to."}},
                "required": ["target_name"],
            },
            handler=handler,
        )

    def _resolve(self, target: str) -> Agent | None:
        """Exact name match first; falls back to a case-insensitive match,
        since the LLM won't always reproduce a name's exact capitalization.
        """
        agent = self.get(target)
        if agent is not None:
            return agent
        lowered = target.strip().lower()
        for candidate in self._agents.values():
            if candidate.identity.name.lower() == lowered:
                return candidate
        return None
