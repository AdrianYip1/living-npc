from __future__ import annotations

import threading
import time
from pathlib import Path

from .agent import Agent
from .conversation import run_conversation, save_transcript
from .identity import DEFAULT_PROFILE_TEMPLATE
from .inventory import TradeError, trade
from .llm import LLMClient
from .storage import (
    load_identities,
    load_instructions,
    load_inventory,
    load_memory,
    load_profile_template,
    save_inventory,
    save_memory,
)
from .tools import Tool, ToolRegistry, make_wait_tool
from .world import INTERACTION_RANGE, clamp_coordinate, distance, step_toward

# initiate_conversation's result on success -- every refusal (unknown name,
# out of range, busy) returns something else. See conversation_happened().
CONVERSATION_HAPPENED_PREFIX = "You had a conversation with "


def conversation_happened(tool_result: object) -> bool:
    """Whether an initiate_conversation call actually ran an exchange, for
    callers (the world tick) that only see the tool's result string.
    """
    return isinstance(tool_result, str) and tool_result.startswith(CONVERSATION_HAPPENED_PREFIX)


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
        inventory_dir: str | Path | None = None,
    ) -> None:
        self._memory_dir = Path(memory_dir)
        # None means inventories aren't persisted -- every run starts from
        # each identity's starting_money / starting_items.
        self._inventory_dir = Path(inventory_dir) if inventory_dir else None
        # The world tick thread and the player-conversation HTTP thread can
        # both trigger trades; a trade touches two NPCs' inventories, so it
        # has to be one atomic step across both.
        self._trade_lock = threading.Lock()
        self._conversation_log_dir = Path(conversation_log_dir) if conversation_log_dir else None
        self._busy: set[str] = set()
        instructions = load_instructions(instructions_path) if instructions_path else ""
        profile_template = (load_profile_template(instructions_path) if instructions_path else "") or DEFAULT_PROFILE_TEMPLATE
        common_tools = tools.all_tools() if tools is not None else []

        self._agents: dict[str, Agent] = {}
        for identity in load_identities(npcs_path):
            memory = load_memory(identity.name, self._memory_dir)
            inventory = load_inventory(identity.name, self._inventory_dir) if self._inventory_dir else None
            npc_tools = ToolRegistry()
            for tool in common_tools:
                npc_tools.register(tool)
            npc_tools.register(self._make_initiate_conversation_tool(identity.name, conversation_turns))
            npc_tools.register(self._make_move_tool(identity.name))
            npc_tools.register(make_wait_tool())
            npc_tools.register(self._make_check_inventory_tool(identity.name))
            npc_tools.register(self._make_trade_tool(identity.name, buying=True))
            npc_tools.register(self._make_trade_tool(identity.name, buying=False))
            self._agents[identity.name] = Agent(
                identity,
                llm,
                npc_tools,
                memory=memory,
                instructions=instructions,
                profile_template=profile_template,
                position=identity.home,
                inventory=inventory,
            )

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    def all(self) -> list[Agent]:
        return list(self._agents.values())

    def save_all(self) -> None:
        for agent in self._agents.values():
            save_memory(agent.identity.name, agent.memory, self._memory_dir)
            if self._inventory_dir is not None:
                save_inventory(agent.identity.name, agent.inventory, self._inventory_dir)

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

    def try_occupy(self, name: str) -> bool:
        """Single-participant version of try_occupy_pair -- for a
        conversation with the player, who isn't an NPC in this registry and
        so has no second name to pair against. Reuses the same busy set, so
        an NPC talking to the player is exactly as unavailable for
        NPC-to-NPC conversation or a world tick as one already mid-exchange.
        """
        return self.try_occupy_pair(name, name)

    def release(self, name: str) -> None:
        self.release_pair(name, name)

    # ------------------------------------------------------------------ #
    # movement
    # ------------------------------------------------------------------ #
    def step_movement(self, dt: float) -> None:
        """Advances every NPC's walk by `dt` real seconds. A busy NPC (mid-
        conversation) brakes to a stop where it is, but keeps its
        destination and resumes walking once it's free again.
        """
        for agent in self._agents.values():
            busy = self.is_busy(agent.identity.name)
            if agent.destination is None and agent.velocity == (0.0, 0.0):
                continue
            position, velocity, arrived = step_toward(
                agent.position, agent.velocity, None if busy else agent.destination, dt
            )
            agent.position, agent.velocity = position, velocity
            if arrived:
                agent.destination = None

    # ------------------------------------------------------------------ #
    def _make_initiate_conversation_tool(self, initiator_name: str, turns: int) -> Tool:
        def handler(target_name: str) -> str:
            target = self._resolve(target_name)
            if target is None:
                return f"There's no one named {target_name!r} nearby."
            resolved_name = target.identity.name
            too_far = self._range_error(initiator_name, target, "talk")
            if too_far is not None:
                return too_far
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
            return f"{CONVERSATION_HAPPENED_PREFIX}{resolved_name}."

        return Tool(
            name="initiate_conversation",
            description=(
                "Start a conversation with another NPC. Only works if they're within "
                f"{INTERACTION_RANGE} units of you -- walk over to them first if they aren't."
            ),
            parameters={
                "type": "object",
                "properties": {"target_name": {"type": "string", "description": "The name of the NPC to talk to."}},
                "required": ["target_name"],
            },
            handler=handler,
        )

    def _make_move_tool(self, name: str) -> Tool:
        def handler(x: int, y: int) -> str:
            clamped = (clamp_coordinate(int(x)), clamp_coordinate(int(y)))
            self._agents[name].destination = clamped
            return f"You start walking to ({clamped[0]}, {clamped[1]})."

        return Tool(
            name="move_to",
            description=(
                "Start walking to a specific spot on the map. You don't arrive instantly -- it "
                "takes time to cross the map. Coordinates run from -100 to 100 on both axes; "
                "anything outside that range is clamped to the nearest edge."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Target x coordinate, from -100 to 100."},
                    "y": {"type": "integer", "description": "Target y coordinate, from -100 to 100."},
                },
                "required": ["x", "y"],
            },
            handler=handler,
        )

    def _make_check_inventory_tool(self, name: str) -> Tool:
        def handler() -> str:
            return self._agents[name].inventory.describe()

        return Tool(
            name="check_inventory",
            description="Look through what you're carrying: your coins and every item you have.",
            parameters={"type": "object", "properties": {}},
            handler=handler,
        )

    def _make_trade_tool(self, name: str, *, buying: bool) -> Tool:
        """buy_item and sell_item are the same trade seen from opposite
        sides -- `name` is the buyer for one and the seller for the other.
        The inventory check itself is all in inventory.trade(); this only
        resolves the counterparty, runs the trade under the lock, and on
        success records it in the counterparty's memory too (the caller's
        own memory already gets the tool result via Agent.respond()).
        """
        counterparty_field = "seller_name" if buying else "buyer_name"

        def handler(item: str, total_price: int, quantity: int = 1, **kwargs: str) -> str:
            counterparty_arg = kwargs.get(counterparty_field, "")
            other = self._resolve(counterparty_arg)
            if other is None:
                return f"There's no one named {counterparty_arg!r} to trade with."
            other_name = other.identity.name
            buyer_name, seller_name = (name, other_name) if buying else (other_name, name)
            quantity, total_price = int(quantity), int(total_price)

            with self._trade_lock:
                # Checked inside the lock alongside the inventories, so the
                # whole trade is judged against one consistent snapshot.
                too_far = self._range_error(name, other, "trade")
                if too_far is not None:
                    return f"The trade didn't go through: {too_far}"
                try:
                    trade(
                        buyer=self._agents[buyer_name].inventory,
                        seller=self._agents[seller_name].inventory,
                        item=item,
                        quantity=quantity,
                        total_price=total_price,
                        buyer_name=buyer_name,
                        seller_name=seller_name,
                    )
                except TradeError as e:
                    return f"The trade didn't go through: {e}"

            goods, price = f"{quantity} x {item}", f"for {total_price} coins"
            if buying:
                mine, theirs = f"You bought {goods} from {other_name} {price}.", f"Sold {goods} to {name} {price}."
            else:
                mine, theirs = f"You sold {goods} to {other_name} {price}.", f"Bought {goods} from {name} {price}."
            other.memory.add(theirs, importance=7, tags={name})
            return f"{mine} {self._agents[name].inventory.describe()}"

        if buying:
            tool_name, verb, counterparty_desc = "buy_item", "Buy an item from", "The name of the NPC you're buying from."
            requirement = "they actually have the item, you actually have the coins,"
        else:
            tool_name, verb, counterparty_desc = "sell_item", "Sell one of your items to", "The name of the NPC you're selling to."
            requirement = "you actually have the item, they actually have the coins,"

        return Tool(
            name=tool_name,
            description=(
                f"{verb} another NPC for an agreed total price. Only goes through if {requirement} "
                f"and they're within {INTERACTION_RANGE} units of you -- agree on the price out loud first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "The name of the item being traded."},
                    counterparty_field: {"type": "string", "description": counterparty_desc},
                    "total_price": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Total coins for the whole deal (not per item).",
                    },
                    "quantity": {"type": "integer", "minimum": 1, "description": "How many of the item. Defaults to 1."},
                },
                "required": ["item", counterparty_field, "total_price"],
            },
            handler=handler,
        )

    def _range_error(self, name: str, other: Agent, verb: str) -> str | None:
        """The one proximity rule every NPC-to-NPC interaction (talking,
        trading) goes through: None if `other` is within INTERACTION_RANGE
        of `name`, otherwise a message to hand back as the tool result.
        """
        gap = distance(self._agents[name].position, other.position)
        if gap <= INTERACTION_RANGE:
            return None
        return f"{other.identity.name} is {gap:.0f} units away -- you need to be within {INTERACTION_RANGE} to {verb}."

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

    def resolve(self, target: str) -> str | None:
        """Public, name-only version of _resolve() -- lets a caller outside
        the registry (the world tick loop) find out who an
        initiate_conversation call actually landed on, by canonical name,
        without reaching into a private method or an Agent it doesn't
        otherwise need.
        """
        agent = self._resolve(target)
        return agent.identity.name if agent is not None else None
