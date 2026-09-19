from __future__ import annotations

import threading
import time
from pathlib import Path

from .agent import Agent
from .conversation import ConversationHooks, run_conversation, save_transcript
from .identity import DEFAULT_PROFILE_TEMPLATE
from .inventory import TradeError, trade
from .llm import LLMClient
from .identity import Identity
from .memory import MemoryStore
from .storage import (
    load_identities,
    load_instructions,
    load_inventory,
    load_memory,
    load_places,
    load_profile_template,
    render_places,
    save_inventory,
    save_memory,
)
from .tools import Tool, ToolRegistry, make_wait_tool
from .world import (
    HESITATE_SPEED_FACTOR,
    INTERACTION_RANGE,
    NPC_MAX_SPEED,
    clamp_coordinate,
    distance,
    keep_personal_space,
    step_toward,
)

# initiate_conversation's result on success -- the first when the exchange
# ran inline, the second when it was handed off to ConversationHooks.run and
# is still going. Every refusal (unknown name, out of range, busy) returns
# something else. See conversation_happened().
CONVERSATION_HAPPENED_PREFIX = "You had a conversation with "
CONVERSATION_STARTED_PREFIX = "You start a conversation with "


def conversation_happened(tool_result: object) -> bool:
    """Whether an initiate_conversation call actually started an exchange
    (finished or still running), for callers (the world tick) that only see
    the tool's result string.
    """
    return isinstance(tool_result, str) and tool_result.startswith(
        (CONVERSATION_HAPPENED_PREFIX, CONVERSATION_STARTED_PREFIX)
    )


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

    Travelers (add_traveler / remove_traveler) are live Agents here too, so
    residents can talk to and trade with them by name exactly like with
    each other -- but they come and go at runtime, get a deliberately
    small action set (move, start a conversation, buy, wait) and a hard
    cap on how much they say, and are never saved to disk. Because agents
    can now appear and disappear while other threads are mid-iteration,
    every walk over the roster goes through a snapshot (all()).
    """

    # Hard cap on a traveler's spoken lines, in words -- enforced by Agent
    # (see max_utterance_words), and quoted to the traveler in its prompt.
    TRAVELER_MAX_WORDS = 12

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
        world_path: str | Path | None = None,
    ) -> None:
        # Exposed so anything generating content in the same world (e.g.
        # traveler identities) uses the same backend as the NPCs.
        self.llm = llm
        self._conversation_turns = conversation_turns
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
        # Who each NPC in an NPC-to-NPC conversation is talking to, both
        # ways round. Guarded by _busy_lock along with _busy.
        self._partners: dict[str, str] = {}
        # How exchanges are run, paced, and watched -- see ConversationHooks.
        # Left at the defaults (inline, unwatched) unless someone like
        # mini_map.Simulation fills it in.
        self.conversation_hooks = ConversationHooks()
        # Guards _busy and roster changes together, so "is this traveler
        # free to leave?" and "claim them for a conversation" can't race.
        self._busy_lock = threading.RLock()
        instructions = load_instructions(instructions_path) if instructions_path else ""
        profile_template = (load_profile_template(instructions_path) if instructions_path else "") or DEFAULT_PROFILE_TEMPLATE
        traveler_lines = load_instructions(instructions_path, key="traveler_instructions") if instructions_path else ""
        traveler_lines = traveler_lines.replace("{max_words}", str(self.TRAVELER_MAX_WORDS))
        self._traveler_instructions = "\n".join(part for part in (instructions, traveler_lines) if part)
        self._traveler_profile_template = (
            load_profile_template(instructions_path, key="traveler_profile_template") if instructions_path else ""
        ) or profile_template
        # Shared by everyone, residents and travelers alike (see world.json).
        # Public so the simulation can pick a place for a traveler to visit.
        self.places = load_places(world_path) if world_path else []
        self._world_context = render_places(self.places)
        common_tools = tools.all_tools() if tools is not None else []

        self._agents: dict[str, Agent] = {}
        self._travelers: set[str] = set()
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
                standing_context=self._world_context,
            )

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    def all(self) -> list[Agent]:
        """Residents and travelers, as a snapshot -- safe to iterate while
        travelers arrive or leave on other threads.
        """
        with self._busy_lock:
            return list(self._agents.values())

    def residents(self) -> list[Agent]:
        return [agent for agent in self.all() if agent.identity.name not in self._travelers]

    def travelers(self) -> list[Agent]:
        return [agent for agent in self.all() if agent.identity.name in self._travelers]

    def is_traveler(self, name: str) -> bool:
        return name in self._travelers

    # ------------------------------------------------------------------ #
    # travelers
    # ------------------------------------------------------------------ #
    def add_traveler(self, identity: Identity, *, position: tuple[float, float], standing_context: str = "") -> Agent:
        """Brings a traveler into the world as a live Agent at `position`.
        The name must not already be taken (resident or traveler) -- pick
        one with unique_name() first.
        """
        name = identity.name
        tools = ToolRegistry()
        tools.register(self._make_move_tool(name))
        tools.register(self._make_initiate_conversation_tool(name, self._conversation_turns))
        tools.register(self._make_trade_tool(name, buying=True))
        tools.register(make_wait_tool())
        agent = Agent(
            identity,
            self.llm,
            tools,
            memory=MemoryStore(),
            instructions=self._traveler_instructions,
            profile_template=self._traveler_profile_template,
            position=position,
            standing_context="\n\n".join(part for part in (standing_context, self._world_context) if part),
            max_utterance_words=self.TRAVELER_MAX_WORDS,
        )
        with self._busy_lock:
            if self._resolve(name) is not None:
                raise ValueError(f"an NPC named {name!r} is already here")
            self._agents[name] = agent
            self._travelers.add(name)
        return agent

    def remove_traveler(self, name: str) -> bool:
        """Takes a traveler out of the world. Refuses (False) while they're
        busy, so nobody leaves mid-conversation; also takes the trade lock,
        so nobody leaves mid-trade either.
        """
        with self._trade_lock, self._busy_lock:
            if name not in self._travelers or name in self._busy:
                return False
            self._travelers.discard(name)
            del self._agents[name]
            return True

    def unique_name(self, name: str) -> str:
        """`name`, or `name` with a number appended if someone here already
        has it (case-insensitively, the same way _resolve() matches).
        """
        candidate, n = name, 2
        while self._resolve(candidate) is not None:
            candidate, n = f"{name} {n}", n + 1
        return candidate

    def save_all(self) -> None:
        for agent in self.residents():
            save_memory(agent.identity.name, agent.memory, self._memory_dir)
            if self._inventory_dir is not None:
                save_inventory(agent.identity.name, agent.inventory, self._inventory_dir)

    # ------------------------------------------------------------------ #
    # busy tracking
    # ------------------------------------------------------------------ #
    def is_busy(self, name: str) -> bool:
        return name in self._busy

    def partner_of(self, name: str) -> str | None:
        """Who `name` is in an NPC-to-NPC conversation with right now, if
        anyone. None while talking to the player -- see try_occupy().
        """
        return self._partners.get(name)

    def try_occupy_pair(self, a_name: str, b_name: str) -> bool:
        """Claims both names at once, or neither. Also what stops a
        conversation from re-initiating mid-exchange: a side already busy
        (both participants are marked busy for the whole exchange, including
        the initiator) can never successfully claim a new pair, so a nested
        initiate_conversation call just fails harmlessly instead of
        recursing.
        """
        with self._busy_lock:
            if a_name not in self._agents or b_name not in self._agents:
                return False  # one of them has already left town
            if self.is_busy(a_name) or self.is_busy(b_name):
                return False
            self._busy.add(a_name)
            self._busy.add(b_name)
            if a_name != b_name:
                self._partners[a_name], self._partners[b_name] = b_name, a_name
            return True

    def release_pair(self, a_name: str, b_name: str) -> None:
        with self._busy_lock:
            self._busy.discard(a_name)
            self._busy.discard(b_name)
            self._partners.pop(a_name, None)
            self._partners.pop(b_name, None)

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
    def step_movement(self, dt: float, hesitating: set[str] | frozenset[str] = frozenset()) -> None:
        """Advances every NPC's walk by `dt` real seconds. A busy NPC (mid-
        conversation) brakes to a stop where it is, but keeps its
        destination and resumes walking once it's free again. Anyone named
        in `hesitating` -- e.g. a traveler deciding what to do about someone
        it just noticed -- slows right down (HESITATE_SPEED_FACTOR) instead
        of walking on past at full speed; stopping dead looked like a stall
        whenever it then just carried on.
        """
        for agent in self.all():
            busy = self.is_busy(agent.identity.name)
            if agent.destination is None and agent.velocity == (0.0, 0.0):
                continue
            max_speed = NPC_MAX_SPEED * (HESITATE_SPEED_FACTOR if agent.identity.name in hesitating else 1.0)
            position, velocity, arrived = step_toward(
                agent.position, agent.velocity, None if busy else agent.destination, dt, max_speed
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
            initiator = self.get(initiator_name)
            hooks = self.conversation_hooks

            def exchange() -> None:
                try:
                    transcript = run_conversation(
                        initiator, target, turns=turns, scene=hooks.scene, on_turn=hooks.on_turn
                    )
                    if self._conversation_log_dir is not None:
                        log_path = (
                            self._conversation_log_dir
                            / f"{initiator_name}_{resolved_name}_{int(time.time() * 1000)}.json"
                        )
                        save_transcript(transcript, log_path)
                    if hooks.on_end is not None:
                        hooks.on_end(transcript)
                finally:
                    self.release_pair(initiator_name, resolved_name)

            if hooks.run is None:
                exchange()
                return f"{CONVERSATION_HAPPENED_PREFIX}{resolved_name}."
            if not hooks.run(exchange):
                self.release_pair(initiator_name, resolved_name)
                return f"You can't talk to {resolved_name} right now."
            return f"{CONVERSATION_STARTED_PREFIX}{resolved_name}."

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
            agent = self._agents.get(name)
            if agent is None:
                return "You've already left town."
            clamped = (clamp_coordinate(int(x)), clamp_coordinate(int(y)))
            others = [
                point
                for other in self.all()
                if other is not agent
                for point in (other.position, other.destination)
                if point is not None
            ]
            destination = keep_personal_space(clamped, agent.position, others)
            agent.destination = destination
            return f"You start walking to ({destination[0]}, {destination[1]})."

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
                # A traveler can only leave under this same lock (see
                # remove_traveler), so once both are here, they stay here.
                if buyer_name not in self._agents or seller_name not in self._agents:
                    return f"The trade didn't go through: {other_name} isn't here anymore."
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
        for candidate in self.all():
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
