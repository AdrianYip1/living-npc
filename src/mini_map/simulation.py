"""Ties the environment agent's tick to the NPC registry -- the piece that
makes launching the mini-map "activate" the world instead of just drawing
it. Lives here (not in game_agents or environment_agent) so neither agent
package has to know the other exists; this is the only thing that imports
both.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from typing import Any

from environment_agent.agent import EnvironmentAgent
from environment_agent.time_of_day import MINUTES_PER_DAY
from environment_agent.travelers import Traveler

from game_agents.agent import Agent, Scene, TurnResult
from game_agents.conversation import ConversationHooks, ConversationTurn
from game_agents.conversation_export import PLAYER_PARTICIPANT, ConversationExporter, npc_participant
from game_agents.identity import Identity
from game_agents.llm import SPEAK_TOOL_NAME
from game_agents.registry import NPCRegistry, conversation_happened
from game_agents.storage import identity_to_record
from game_agents.traveler_identity import generate_traveler_identity
from game_agents.world import INTERACTION_RANGE, MAP_MAX, MAP_MIN, distance, render_surroundings

from .traveler_purpose import BUY, TravelerPurpose, pick_purpose

log = logging.getLogger(__name__)

# Nobody speaks outside a conversation. Given the chance, travelers
# announced what they were about to do ("Time I got moving!") instead of
# doing it -- a turn spent speaking is a turn not spent walking -- or
# greeted someone they hadn't started a conversation with, and residents
# said the same line to themselves every few turns ("Hinges won't hammer
# themselves"), sometimes aimed at whoever they'd just said goodbye to.
# To talk to someone, you start a conversation with them.
NO_SPEAKING = frozenset({SPEAK_TOOL_NAME})
# Anyone's first turn after a conversation: the goodbye's been said, so no
# saying it again -- and no starting straight back up with whoever's still
# standing there, which is what a model does once it can't speak.
AFTER_CONVERSATION = NO_SPEAKING | {"initiate_conversation"}


class Simulation:
    """Two independently-paced real-time cadences, decoupled on purpose, and
    configured as rates rather than raw intervals so they read the way
    you'd actually reason about them:

    - ticks_per_real_minute: how fast the in-game clock itself runs -- how
      often the environment agent advances weather/time-of-day/travelers
      (see its own docstring; still deterministic, no LLM call of its own
      yet).
    - llm_calls_per_game_hour: how often idle NPCs actually get stimulated,
      i.e. how often their LLM gets queried -- expressed against *game*
      time (e.g. "each NPC thinks about 4 times an hour") rather than real
      time, since that's the number that actually describes how NPCs feel,
      independent of how fast you've got the clock itself running. This is
      the expensive/slow part, so it's allowed to lag behind a fast-ticking
      clock instead of firing every single tick.

    Both get converted once, at construction, into the real-second
    intervals the two background loops actually wait on -- llm_calls_per_
    real_minute (also exposed, in state(), purely for display: "given your
    tick speed and your game-hour setting, here's what that costs in real
    API calls per minute") is the one live combination of the two.

    run_forever() runs each on its own daemon thread rather than
    interleaving them on one -- a real LLM backend can take several real
    seconds per NPC, and if that ran on the same thread/loop as the
    environment tick, a slow query round would stall the "fast" clock too,
    defeating the point of decoupling the two rates in the first place.
    tick() is the synchronous, do-both-right-now version -- what tests and
    any other direct caller use for a deterministic beat.

    A third loop steps NPC walking physics (NPCRegistry.step_movement) at
    MOVEMENT_HZ in *real* time, not game time -- NPCs walk at the same
    on-screen speed as the player no matter how fast the clock runs.

    Traveler arrivals come from the environment agent as a short sketch;
    each is expanded into a full Identity (npcs.json format) by the LLM on
    a small worker pool, not the environment thread -- same reasoning as
    above: a slow LLM call must not stall the clock. The traveler then
    joins the registry as a live Agent at a random point on a map edge,
    told (in its system prompt) that its goal is an exit point on the
    opposite edge, and why it stopped in town at all: a TravelerPurpose
    rolled on arrival -- just passing through, spending some time at one of
    world.json's places, or buying something from the resident who works
    at one (see traveler_purpose.py).

    Travelers aren't on the residents' llm_calls_per_game_hour cadence --
    at NPC walking speed they cross the whole map in seconds, well inside
    one query round, so they'd never get a chance to be distracted.
    Instead _update_travelers() (run from the movement loop, and cheap: it
    never calls the LLM itself) gives each one a turn when something
    happens: on arrival, when someone it hasn't noticed yet comes within
    TRAVELER_NOTICE_RANGE, when it reaches wherever it was walking to, and
    when it's been standing idle a while. Any of
    those turns can move_to somewhere else, which is how a traveler
    interrupts its own walk. While a turn is being decided the traveler
    slows to a hesitant walk (HESITATE_SPEED_FACTOR), rather than striding
    on past whatever it noticed -- or freezing, which read as a stall. It
    leaves town (is removed) on reaching its exit point, and heads there
    for good if it waits TRAVELER_MAX_WAIT_MINUTES on someone busy or
    stays TRAVELER_MAX_STAY_MINUTES in all.

    NPC-to-NPC conversations run on their own worker pool (see
    ConversationHooks, which this fills in on the registry): the
    initiator's initiate_conversation call returns as soon as the pair is
    claimed, so a long exchange never holds up anyone else's turn. Every
    spoken line -- in a conversation, to the player, or to no one in
    particular -- goes into a speech feed state() exposes, for the page to
    draw as speech bubbles. With line_pacing on, each conversation line
    waits until the previous one has been up long enough to read (see
    _read_seconds()), and the pair stays together until the last one has
    too; pausing freezes that wait, same as the rest of the world.

    Given an `exporter`, every conversation -- NPC-to-NPC and the player's
    -- is also written out for the speech/animation program as it happens
    (see game_agents.conversation_export), each line at the moment its
    bubble goes up. active.json points that program at the player's own
    conversation if there is one, otherwise at the NPC-to-NPC one nearest
    the player (whose position the page reports -- see
    set_player_position()).
    """

    MOVEMENT_HZ = 30.0
    # How many recent traveler arrivals state() keeps around. The page
    # spawns each one once, by id, so this only needs to cover arrivals
    # between two polls -- the rest is slack for a slow/backgrounded tab.
    RECENT_TRAVELERS = 20
    # Cap on one movement step, so a stalled thread (or a long pause
    # resuming) doesn't fling NPCs across the map in a single jump.
    MAX_MOVEMENT_DT_S = 0.1
    # Map units. Wider than INTERACTION_RANGE, so a traveler notices
    # someone early enough to decide to walk over.
    TRAVELER_NOTICE_RANGE = 25
    # Game-minutes a traveler stands idle before it's prompted again.
    TRAVELER_IDLE_TURN_MINUTES = 15
    # Game-minutes before a traveler that still hasn't left is sent to its
    # exit point without asking -- a backstop for a model that never
    # moves, not a script.
    TRAVELER_MAX_STAY_MINUTES = 4 * 60
    # How close counts as having reached the exit point, in map units.
    TRAVELER_EXIT_REACHED = 1.0
    # Game-minutes a traveler waits for someone busy in a conversation
    # before giving up on them and leaving town.
    TRAVELER_MAX_WAIT_MINUTES = 10
    # Coins a traveler carries when its identity comes from the fallback
    # (no LLM), and the least one that came to buy something ever carries.
    TRAVELER_FALLBACK_MONEY = (5, 30)
    TRAVELER_BUYER_MIN_MONEY = 20
    # How long, in game-minutes, before the same two can start another
    # conversation -- otherwise a resident keeps greeting a traveler that's
    # lingering nearby, and they go over the same ground again.
    REPEAT_CONVERSATION_MINUTES = 60
    # How many recent spoken lines state() keeps around -- same slack
    # reasoning as RECENT_TRAVELERS.
    RECENT_SPEECH = 60
    # How long a spoken line stays up, in real seconds: long enough to read
    # at a relaxed pace, within these bounds. Also how long a conversation
    # waits before its next line (see _on_conversation_turn).
    SECONDS_PER_WORD = 0.3
    MIN_LINE_SECONDS = 2.5
    MAX_LINE_SECONDS = 7.0

    def __init__(
        self,
        registry: NPCRegistry,
        environment: EnvironmentAgent,
        *,
        ticks_per_real_minute: float = 60.0,
        llm_calls_per_game_hour: float = 4.0,
        seed: int | None = None,
        line_pacing: bool = True,
        exporter: ConversationExporter | None = None,
        backend: str = "",
    ) -> None:
        self._registry = registry
        # Which LLM backend is driving the NPCs (bootstrap's GAME_AGENTS_LLM
        # name) -- only reported to the page, so a mock run is obvious.
        self._backend = backend
        self._environment = environment
        # The in-game day residents were last restocked on (see
        # _advance_environment). Starts as the current day, restocked now:
        # saved inventories carry over between runs, so a resident who sold
        # out last session would otherwise stay empty until midnight.
        self._restocked_day = environment.elapsed_minutes // MINUTES_PER_DAY
        registry.restock_residents()
        self._ticks_per_real_minute = ticks_per_real_minute
        self._llm_calls_per_game_hour = llm_calls_per_game_hour
        self._tick_interval_s = 60.0 / ticks_per_real_minute
        self._npc_query_interval_s = 60.0 / self._llm_calls_per_real_minute()
        self._stop = threading.Event()
        # Set while paused -- checked by both loops below, not the player-
        # conversation methods (start_conversation/end_conversation/say):
        # pausing freezes the autonomous world, not the player's own direct
        # interaction with it.
        self._paused = threading.Event()
        # Read by the HTTP handler thread, written by the tick thread below;
        # plain dict/tuple assignment is fine to read concurrently under the
        # GIL for a display that's allowed to be a beat stale.
        self._last_activity: dict[str, str] = {}
        # Travelers the environment agent sent in, tagged with an ever-
        # increasing id so the page can tell new arrivals from ones it has
        # already spawned. Appended by the identity worker threads, copied
        # out by state() on the HTTP thread -- hence the lock.
        self._traveler_arrivals: list[dict[str, Any]] = []
        self._next_traveler_id = 1
        self._traveler_lock = threading.Lock()
        self._identity_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="traveler-identity")
        self._turn_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="traveler-turn")
        self._conversation_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="conversation")
        # Background work (identity generation, traveler turns,
        # conversations) not yet finished -- see wait_for_pending().
        self._pending: list[Future] = []
        self._pending_lock = threading.Lock()
        # Per-traveler bookkeeping, keyed by name, and which travelers
        # have a turn in flight right now (at most one each).
        self._traveler_state: dict[str, _TravelerState] = {}
        self._turns_in_flight: set[str] = set()
        self._turns_lock = threading.Lock()
        # Residents just out of a conversation -- their next turn gets
        # AFTER_CONVERSATION (a traveler's gets it via _TravelerState.talked_with).
        self._just_talked: set[str] = set()
        # When each pair (keyed like _line_ready_at) last finished talking,
        # in elapsed game-minutes -- see REPEAT_CONVERSATION_MINUTES.
        self._talked_at: dict[frozenset[str], int] = {}
        self._rng = random.Random(seed)
        # Spoken lines for the page, tagged with ever-increasing ids like
        # _traveler_arrivals. Appended from whichever thread the line was
        # spoken on.
        self._speech: list[dict[str, Any]] = []
        self._next_speech_id = 1
        self._speech_lock = threading.Lock()
        self._line_pacing = line_pacing
        # Per conversation (keyed by the pair), the real monotonic time its
        # latest line has been up long enough to read.
        self._line_ready_at: dict[frozenset[str], float] = {}
        # Conversations being exported (see the class docstring): NPC-to-NPC
        # ones by pair, as (conversation id, [initiator, target]), and the
        # player's as (conversation id, npc name).
        self._exporter = exporter
        self._exports: dict[frozenset[str], tuple[str, list[str]]] = {}
        self._player_export: tuple[str, str] | None = None
        self._export_lock = threading.Lock()
        self._player_position: tuple[float, float] = (0.0, 0.0)
        registry.conversation_hooks = ConversationHooks(
            run=self._run_conversation,
            on_start=self._on_conversation_start,
            scene=self._scene,
            on_turn=self._on_conversation_turn,
            on_end=self._on_conversation_end,
            refuse=self._refuse_conversation,
        )

    def run_forever(self) -> None:
        env_thread = threading.Thread(target=self._run_environment_loop, daemon=True)
        npc_thread = threading.Thread(target=self._run_npc_query_loop, daemon=True)
        move_thread = threading.Thread(target=self._run_movement_loop, daemon=True)
        env_thread.start()
        npc_thread.start()
        move_thread.start()
        env_thread.join()
        npc_thread.join()
        move_thread.join()

    def stop(self) -> None:
        self._stop.set()
        self._identity_pool.shutdown(wait=False, cancel_futures=True)
        self._turn_pool.shutdown(wait=False, cancel_futures=True)
        self._conversation_pool.shutdown(wait=False, cancel_futures=True)

    def wait_for_pending(self, timeout: float = 10.0) -> None:
        """Blocks until all background work is done -- traveler identity
        generation, any turns that queued (e.g. the arrival turn each
        admitted traveler gets), and conversations, including any those
        started along the way. For tests and other callers that need it all
        to have landed before they look.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._pending_lock:
                self._pending = [f for f in self._pending if not f.done()]
                pending = list(self._pending)
            if not pending:
                return
            wait_futures(pending, timeout=max(0.0, deadline - time.monotonic()))

    def _submit(self, pool: ThreadPoolExecutor, fn, *args) -> bool:
        try:
            future = pool.submit(fn, *args)
        except RuntimeError:  # pool already shut down by stop()
            return False
        with self._pending_lock:
            self._pending = [f for f in self._pending if not f.done()]
            self._pending.append(future)
        return True

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def _run_environment_loop(self) -> None:
        while not self._stop.is_set():
            if not self._paused.is_set():
                self._advance_environment()
            self._stop.wait(self._tick_interval_s)

    def _run_npc_query_loop(self) -> None:
        while not self._stop.is_set():
            if not self._paused.is_set():
                self._query_npcs()
            self._stop.wait(self._npc_query_interval_s)

    def _run_movement_loop(self) -> None:
        last = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            dt = min(now - last, self.MAX_MOVEMENT_DT_S)
            last = now
            if not self._paused.is_set():
                self._registry.step_movement(dt, hesitating=set(self._turns_in_flight))
                self._update_travelers()
            # Even while paused: the player can still talk to someone.
            self.update_active_conversation()
            self._stop.wait(1.0 / self.MOVEMENT_HZ)

    def tick(self) -> None:
        """Force one full beat right now -- advance the environment and
        query every idle NPC, regardless of npc_query_interval_s. Not used
        by run_forever() (which paces the two separately); this is for
        tests and any other caller that wants an immediate, deterministic
        beat instead of waiting on the real-time schedule.
        """
        self._advance_environment()
        self._update_travelers()
        self._query_npcs()

    def _advance_environment(self) -> None:
        event = self._environment.tick()
        day = self._environment.elapsed_minutes // MINUTES_PER_DAY
        if day != self._restocked_day:
            # A new in-game day: the residents' shelves refill.
            self._restocked_day = day
            self._registry.restock_residents()
        for traveler in event.travelers_arrived:
            if not self._submit(self._identity_pool, self._admit_traveler, traveler, event.clock):
                return

    # ------------------------------------------------------------------ #
    # travelers
    # ------------------------------------------------------------------ #
    def _admit_traveler(self, sketch: Traveler, clock: str) -> None:
        """Worker thread: generate the identity, then bring the traveler
        into the world and give it its first turn.
        """
        with self._traveler_lock:
            recent_names = [arrival["identity"]["name"] for arrival in self._traveler_arrivals]
            purpose = pick_purpose(self._rng, self._registry.places, self._registry.residents())
            fallback_money = self._rng.randint(*self.TRAVELER_FALLBACK_MONEY)
        identity = generate_traveler_identity(
            self._registry.llm,
            _traveler_brief(sketch, purpose),
            fallback=_fallback_identity(sketch, purpose, money=fallback_money),
            taken_names=[agent.identity.name for agent in self._registry.all()] + recent_names,
        )
        # The fallback skips the taken-name check, and two generations can
        # race to the same name -- the registry has the final say.
        identity.name = self._registry.unique_name(identity.name)
        if purpose.kind == BUY:
            # Came to buy something: make sure they can pay for it.
            identity.starting_money = max(identity.starting_money, self.TRAVELER_BUYER_MIN_MONEY)

        entry, exit_point = self._pick_entry_and_exit()
        standing_context = (
            f"Your exit point: ({exit_point[0]}, {exit_point[1]}), on the edge of the map. "
            "Reaching it means leaving town.\n" + purpose.standing_context()
        )
        agent = self._registry.add_traveler(identity, position=entry, standing_context=standing_context)
        now = self._environment.elapsed_minutes
        self._traveler_state[identity.name] = _TravelerState(
            exit_point=exit_point,
            arrived_minute=now,
            last_turn_minute=now,
            purpose=purpose,
            item_baseline=agent.inventory.count(purpose.item) if purpose.item else 0,
        )

        # The id is assigned only now, once the traveler exists, so ids in
        # state() always appear in increasing order even if two
        # generations finish out of order.
        with self._traveler_lock:
            self._traveler_arrivals.append(
                {
                    "id": self._next_traveler_id,
                    "arrived_at": clock,
                    "identity": identity_to_record(identity),
                    "purpose": {"kind": purpose.kind, "summary": purpose.summary()},
                }
            )
            self._next_traveler_id += 1
            del self._traveler_arrivals[: -self.RECENT_TRAVELERS]

        self._start_traveler_turn(
            identity.name,
            f"You've just arrived at the edge of town, at ({entry[0]}, {entry[1]}). "
            f"Your exit point is ({exit_point[0]}, {exit_point[1]}), on the far side of the map."
            + self._traveler_state[identity.name].reminder(agent, now),
        )

    def _pick_entry_and_exit(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """A random point on a random edge, and a random point on the
        opposite edge -- so the natural route crosses through town.
        """
        rng = self._rng
        edge = rng.randrange(4)
        a, b = rng.randint(MAP_MIN, MAP_MAX), rng.randint(MAP_MIN, MAP_MAX)
        if edge == 0:  # enters from the top, heads for the bottom
            return (a, MAP_MIN), (b, MAP_MAX)
        if edge == 1:  # right -> left
            return (MAP_MAX, a), (MAP_MIN, b)
        if edge == 2:  # bottom -> top
            return (a, MAP_MAX), (b, MAP_MIN)
        return (MAP_MIN, a), (MAP_MAX, b)  # left -> right

    def _update_travelers(self) -> None:
        """Decides which travelers get a turn right now, and lets go of any
        that reached their exit. Never calls the LLM itself -- turns run on
        _turn_pool -- so it's safe to call from the movement loop.
        """
        now = self._environment.elapsed_minutes
        for name, st in list(self._traveler_state.items()):
            agent = self._registry.get(name)
            if agent is None:
                self._traveler_state.pop(name, None)
                continue
            if name in self._turns_in_flight or self._registry.is_busy(name):
                continue

            if distance(agent.position, st.exit_point) <= self.TRAVELER_EXIT_REACHED:
                if self._registry.remove_traveler(name):
                    self._traveler_state.pop(name, None)
                    self._last_activity.pop(name, None)
                continue

            place = st.purpose.place
            if place is not None and not st.visited and distance(agent.position, place["position"]) <= INTERACTION_RANGE:
                st.visited = True

            if not st.leaving and now - st.arrived_minute >= self.TRAVELER_MAX_STAY_MINUTES:
                st.leaving = True
                self._last_activity[name] = "has lingered long enough and heads out of town"
            if st.leaving:
                if agent.destination != st.exit_point:
                    agent.destination = st.exit_point
                continue

            newly_noticed = [
                other
                for other in self._registry.all()
                if other is not agent
                and other.identity.name not in st.noticed
                and distance(agent.position, other.position) <= self.TRAVELER_NOTICE_RANGE
            ]
            if newly_noticed:
                st.noticed.update(other.identity.name for other in newly_noticed)
                who = ", ".join(
                    f"{other.identity.name}, at ({other.position[0]:.0f}, {other.position[1]:.0f})"
                    for other in newly_noticed
                )
                self._start_traveler_turn(name, f"You notice someone nearby: {who}.")
                continue

            if agent.destination is not None:
                st.heading_to = agent.destination
            elif st.heading_to is not None:
                # Just got where it was going (and it wasn't the exit):
                # decide what's next now, not after standing idle a while.
                reached, st.heading_to = st.heading_to, None
                within_reach = [
                    other.identity.name
                    for other in self._registry.all()
                    if other is not agent and distance(agent.position, other.position) <= INTERACTION_RANGE
                ]
                company = f" Close enough to talk to: {', '.join(within_reach)}." if within_reach else ""
                self._start_traveler_turn(
                    name,
                    f"You've reached ({reached[0]}, {reached[1]}).{company} "
                    f"Your exit point is ({st.exit_point[0]}, {st.exit_point[1]})." + st.reminder(agent, now),
                )
                continue

            if st.waiting_for is not None:
                if not self._registry.is_busy(st.waiting_for):
                    # Whoever it wanted to talk to is free now: say so,
                    # rather than leaving it to hover and try again at random.
                    target, st.waiting_for = st.waiting_for, None
                    if self._registry.get(target) is not None:
                        self._start_traveler_turn(
                            name,
                            f"{target} has finished their conversation and is free to talk now. "
                            f"Your exit point is ({st.exit_point[0]}, {st.exit_point[1]})." + st.reminder(agent, now),
                        )
                        continue
                elif now - st.waiting_since >= self.TRAVELER_MAX_WAIT_MINUTES:
                    # Waited long enough: gives up and leaves town.
                    self._last_activity[name] = f"got tired of waiting for {st.waiting_for} and heads out of town"
                    st.waiting_for, st.leaving = None, True
                    agent.destination = st.exit_point
                    continue

            if st.talked_with is not None:
                # Just out of a conversation: decide what's next now.
                partner, st.talked_with = st.talked_with, None
                self._start_traveler_turn(
                    name,
                    f"You've just finished talking with {partner} and already said your goodbyes. "
                    f"Your exit point is ({st.exit_point[0]}, {st.exit_point[1]})." + st.reminder(agent, now),
                    hide=AFTER_CONVERSATION,
                )
                continue

            idle = agent.destination is None and agent.velocity == (0.0, 0.0)
            if idle and now - st.last_turn_minute >= self.TRAVELER_IDLE_TURN_MINUTES:
                self._start_traveler_turn(
                    name,
                    f"You're standing at ({agent.position[0]:.0f}, {agent.position[1]:.0f}), not walking anywhere. "
                    f"Your exit point is ({st.exit_point[0]}, {st.exit_point[1]})." + st.reminder(agent, now),
                )

    def _start_traveler_turn(self, name: str, stimulus: str, *, hide: frozenset[str] = frozenset()) -> None:
        with self._turns_lock:
            if name in self._turns_in_flight:
                return
            self._turns_in_flight.add(name)
        if not self._submit(self._turn_pool, self._run_traveler_turn, name, stimulus, hide):
            with self._turns_lock:
                self._turns_in_flight.discard(name)

    def _run_traveler_turn(self, name: str, stimulus: str, hide: frozenset[str] = frozenset()) -> None:
        try:
            agent = self._registry.get(name)
            if agent is None:
                return
            destination = agent.destination
            result = agent.respond(stimulus, scene=self._scene(agent), routine=True, hide=NO_SPEAKING | hide)
            if not self._overtaken(agent, result, destination):
                self._record_turn(agent, result)
        except Exception:
            log.exception("traveler %s's turn failed", name)
        finally:
            st = self._traveler_state.get(name)
            if st is not None:
                st.last_turn_minute = self._environment.elapsed_minutes
            with self._turns_lock:
                self._turns_in_flight.discard(name)

    def _scene(self, agent: Agent | None = None) -> Scene:
        """Time and weather -- plus, given the agent whose turn it is, who's
        around them (see world.render_surroundings()).
        """
        clock, phase = self._environment.clock, self._environment.time_of_day.value
        context = f"It's {clock}, {phase}, and {self._environment.weather.value} out."
        if agent is not None:
            context = "\n".join(filter(None, (context, self._surroundings(agent))))
        return Scene(time=f"{clock} ({phase})", context=context)

    def _surroundings(self, agent: Agent) -> str:
        others = [
            (other.identity.name, other.position, self._registry.is_busy(other.identity.name))
            for other in self._registry.all()
            if other is not agent
        ]
        return render_surroundings(agent.position, others)

    def _query_npcs(self) -> None:
        phase = self._environment.time_of_day.value
        weather = self._environment.weather.value
        clock = self._environment.clock

        # An NPC who was pulled into a conversation by someone else's turn
        # this round has already acted -- querying them again would be a
        # second turn in the same beat, not a fresh moment.
        # Residents only -- travelers get their own event-driven turns
        # (see _update_travelers).
        acted: set[str] = set()
        for agent in self._registry.residents():
            name = agent.identity.name
            if name in acted or self._registry.is_busy(name):
                continue

            surroundings = self._surroundings(agent)
            hide = AFTER_CONVERSATION if name in self._just_talked else NO_SPEAKING
            self._just_talked.discard(name)
            scene = Scene(
                time=f"{clock} ({phase})",
                context=(
                    f"It's {clock}, {phase}, and {weather} out. Decide what you do right now, guided by your habits."
                    + (f"\n{surroundings}" if surroundings else "")
                ),
            )
            destination = agent.destination
            result = agent.respond(f"It is now {clock} ({phase}).", scene=scene, routine=True, hide=hide)
            acted.add(name)
            if self._overtaken(agent, result, destination):
                continue
            partner = self._record_turn(agent, result)
            if partner is not None:
                acted.add(partner)

    def _overtaken(self, agent: Agent, result: TurnResult, destination: tuple[int, int] | None) -> bool:
        """Whether someone pulled `agent` into a conversation while this
        turn was still being decided -- it was checked for busy before the
        (slow) LLM call, not after. The decision was made for a moment
        that's gone, so it's dropped: nothing said (it'd land mid-
        conversation, addressed to no one), and no walk queued up for
        afterwards. Its own successful initiate_conversation doesn't count.
        """
        if not self._registry.is_busy(agent.identity.name):
            return False
        action = result.action
        if action is not None and action["name"] == "initiate_conversation" and conversation_happened(action["result"]):
            return False
        if action is not None and action["name"] == "move_to":
            agent.destination = destination
        return True

    def _record_turn(self, agent: Agent, result: TurnResult) -> str | None:
        """Updates the activity line for whoever took a turn -- and, if the
        turn was a conversation that actually happened, for its partner
        too. Returns that partner's name, if any. A refused conversation
        (out of range, busy) leaves the target untouched.
        """
        name = agent.identity.name
        self._publish_trade(name, result.action)
        self._note_busy_refusal(name, result.action)
        if (
            result.action is not None
            and result.action["name"] == "initiate_conversation"
            and conversation_happened(result.action["result"])
        ):
            resolved = self._registry.resolve(result.action["arguments"].get("target_name", ""))
            if resolved is not None:
                st = self._traveler_state.get(name)
                if st is not None:
                    st.waiting_for = None
                # The exchange runs on its own thread and may already be
                # over (and have written "talked with") by now.
                if self._registry.partner_of(name) == resolved:
                    self._last_activity[name] = f"talking with {resolved}"
                    self._last_activity[resolved] = f"talking with {name}"
                return resolved
        self._last_activity[name] = self._describe(result)
        return None

    def _note_busy_refusal(self, name: str, action: dict[str, Any] | None) -> None:
        """A traveler that tried to talk to someone mid-conversation waits
        for them (see _TravelerState.waiting_for).
        """
        st = self._traveler_state.get(name)
        if st is None or action is None or action["name"] != "initiate_conversation":
            return
        result = action["result"]
        if isinstance(result, str) and result.endswith(" is busy right now."):
            target = self._registry.resolve(action["arguments"].get("target_name", ""))
            if target != st.waiting_for:
                # A fresh wait (not a retry of the same one) starts the clock.
                st.waiting_for, st.waiting_since = target, self._environment.elapsed_minutes

    # ------------------------------------------------------------------ #
    # NPC-to-NPC conversations (see ConversationHooks)
    # ------------------------------------------------------------------ #
    def _run_conversation(self, exchange) -> bool:
        def run() -> None:
            try:
                exchange()
            except Exception:
                log.exception("conversation failed")

        return self._submit(self._conversation_pool, run)

    def _on_conversation_start(self, initiator: str, target: str) -> None:
        if self._exporter is None:
            return
        names = [initiator, target]
        agents = [self._registry.get(name) for name in names]
        if None in agents:
            return
        conversation_id = self._exporter.start([npc_participant(agent.identity) for agent in agents])
        with self._export_lock:
            self._exports[frozenset(names)] = (conversation_id, names)

    def _on_conversation_turn(self, turn: ConversationTurn) -> None:
        """Runs on the conversation's worker thread, right after a side
        decides its line: holds it until the previous line has been up long
        enough to read, then publishes it.
        """
        key = frozenset((turn.speaker, turn.listener))
        self._hold_until(self._line_ready_at.get(key, 0.0))
        if turn.utterance is None:
            self._last_activity[turn.speaker] = self._describe(TurnResult(utterance=None, action=turn.action))
            self._publish_trade(turn.speaker, turn.action)
            return
        read_seconds = self._publish_speech(
            turn.speaker, turn.listener, turn.utterance, ends_conversation=turn.ends_conversation
        )
        self._line_ready_at[key] = time.monotonic() + read_seconds
        self._last_activity[turn.speaker] = f"said to {turn.listener}: {turn.utterance}"
        with self._export_lock:
            export = self._exports.get(key)
        speaker = self._registry.get(turn.speaker)
        if export is not None and speaker is not None:
            self._exporter.line(export[0], speaker.identity, turn.utterance)

    def _on_conversation_end(self, transcript: list[ConversationTurn]) -> None:
        """Keeps the pair together until the last line has been read, then
        lets them go (the registry releases them right after this).
        """
        if not transcript:
            return
        a, b = transcript[0].speaker, transcript[0].listener
        self._hold_until(self._line_ready_at.pop(frozenset((a, b)), 0.0))
        self._end_export(frozenset((a, b)))
        self._talked_at[frozenset((a, b))] = self._environment.elapsed_minutes
        self._last_activity[a] = f"talked with {b}"
        self._last_activity[b] = f"talked with {a}"
        if not any(turn.utterance for turn in transcript):
            # Nothing was said -- no conversation to follow up on, and
            # prompting right away again risks a tight loop of non-starts.
            return
        for name, partner in ((a, b), (b, a)):
            st = self._traveler_state.get(name)
            if st is not None:
                st.talked_with = partner
            else:
                self._just_talked.add(name)

    def _end_export(self, key: frozenset[str]) -> None:
        with self._export_lock:
            export = self._exports.pop(key, None)
        if export is not None:
            self._exporter.end(export[0])

    def set_player_position(self, x: float, y: float) -> None:
        self._player_position = (x, y)

    def update_active_conversation(self) -> None:
        """Points active.json at the player's conversation if there is one,
        otherwise the NPC-to-NPC one nearest the player (or at nothing).
        Also ends the export of any NPC-to-NPC conversation whose pair was
        let go without _on_conversation_end running (the exchange failed).
        Cheap -- the file is only rewritten when the answer changes.
        """
        if self._exporter is None:
            return
        with self._export_lock:
            player_export = self._player_export
            exports = dict(self._exports)
        for key, (_, (a, b)) in list(exports.items()):
            if self._registry.partner_of(a) != b:
                self._end_export(key)
                exports.pop(key)
        if player_export is not None:
            self._exporter.point_at(player_export[0], [player_export[1]])
            return
        best: tuple[float, str, list[str]] | None = None
        for conversation_id, names in exports.values():
            agents = [self._registry.get(name) for name in names]
            if None in agents:
                continue
            dist = min(distance(self._player_position, agent.position) for agent in agents)
            if best is None or dist < best[0]:
                best = (dist, conversation_id, names)
        if best is None:
            self._exporter.point_at(None, [])
        else:
            self._exporter.point_at(best[1], best[2])

    def _refuse_conversation(self, initiator: str, target: str) -> str | None:
        talked_at = self._talked_at.get(frozenset((initiator, target)))
        if talked_at is None or self._environment.elapsed_minutes - talked_at >= self.REPEAT_CONVERSATION_MINUTES:
            return None
        return f"You only just talked with {target} -- there's nothing new to say yet. Leave them be for now."

    def _hold_until(self, deadline: float) -> None:
        """Waits out the real time left until `deadline` -- except that time
        spent paused doesn't count, and a paused world holds here even past
        the deadline, so a conversation freezes mid-exchange like everything
        else. A no-op without line_pacing.
        """
        if not self._line_pacing:
            return
        step = 0.05
        remaining = deadline - time.monotonic()
        while not self._stop.is_set() and (remaining > 0 or self._paused.is_set()):
            self._stop.wait(step)
            if not self._paused.is_set():
                remaining -= step

    def _read_seconds(self, text: str) -> float:
        return min(self.MAX_LINE_SECONDS, max(self.MIN_LINE_SECONDS, self.SECONDS_PER_WORD * len(text.split())))

    def _publish_trade(self, name: str, action: dict[str, Any] | None) -> None:
        """A trade that went through goes in the feed too (kind "trade"), as
        a narrated line -- "Wren bought 1 x horseshoe from Mara for 5
        coins." -- so the page can show it happening.
        """
        if action is None or action["name"] not in ("buy_item", "sell_item"):
            return
        result = action["result"]
        if not isinstance(result, str) or not result.startswith(("You bought ", "You sold ")):
            return
        text = f"{name} {result.split('. ', 1)[0].removeprefix('You ').rstrip('.')}."
        self._publish_speech(name, None, text, kind="trade")

    def _publish_speech(
        self,
        speaker: str,
        listener: str | None,
        text: str,
        *,
        ends_conversation: bool = False,
        kind: str = "speech",
    ) -> float:
        """Adds a line to the speech feed and returns how long it should
        stay up. `listener` is another NPC's name, "player", or None for a
        line said to no one in particular. `kind` is "speech" for a spoken
        line, "trade" for a narrated trade (see _publish_trade).
        """
        read_seconds = self._read_seconds(text)
        with self._speech_lock:
            self._speech.append(
                {
                    "id": self._next_speech_id,
                    "kind": kind,
                    "speaker": speaker,
                    "listener": listener,
                    "text": text,
                    "read_seconds": read_seconds,
                    "ends_conversation": ends_conversation,
                    "clock": self._environment.clock,
                }
            )
            self._next_speech_id += 1
            del self._speech[: -self.RECENT_SPEECH]
        return read_seconds

    # ------------------------------------------------------------------ #
    # player conversation (see mini_map.game's /api/conversation/* routes)
    # ------------------------------------------------------------------ #
    def start_conversation(self, name: str) -> bool:
        """Claims the NPC as busy for the duration of a player conversation
        -- same busy set the world tick and NPC-to-NPC conversations check,
        so this NPC is skipped by both for as long as the player's talking
        to them, exactly as if they were mid-exchange with someone else.
        """
        agent = self._registry.get(name)
        if agent is None or not self._registry.try_occupy(name):
            return False
        if self._exporter is not None:
            conversation_id = self._exporter.start([dict(PLAYER_PARTICIPANT), npc_participant(agent.identity)])
            with self._export_lock:
                previous, self._player_export = self._player_export, (conversation_id, name)
            if previous is not None:
                self._exporter.end(previous[0])
        return True

    def end_conversation(self, name: str) -> None:
        with self._export_lock:
            export = self._player_export
            if export is not None and export[1] == name:
                self._player_export = None
            else:
                export = None
        if export is not None:
            self._exporter.end(export[0])
        self._registry.release(name)

    def say(self, name: str, text: str) -> dict[str, Any] | None:
        agent = self._registry.get(name)
        if agent is None:
            return None
        with self._export_lock:
            export = self._player_export
        if export is not None and export[1] != name:
            export = None

        scene = Scene(
            time=f"{self._environment.clock} ({self._environment.time_of_day.value})",
            context=(
                f"The player has walked up and is speaking with you directly. "
                f"It's {self._environment.weather.value} out."
            ),
        )
        # conversation=True: the player is talking with them, so speaking
        # means answering, not thinking aloud (see Agent._speak_schema).
        result = agent.respond(text, scene=scene, tags={"player"}, conversation=True)
        self._last_activity[name] = self._describe(result)
        if result.utterance is not None:
            self._publish_speech(name, "player", result.utterance)
            if export is not None:
                # A no-op if the player already walked off mid-reply.
                self._exporter.line(export[0], agent.identity, result.utterance)
        return {
            "utterance": result.utterance,
            "action": None if result.action is None else {"name": result.action["name"]},
        }

    def _describe(self, result: TurnResult) -> str:
        if result.utterance is not None:
            return f"said: {result.utterance}"
        tool_result = result.action["result"]
        return tool_result if isinstance(tool_result, str) else result.action["name"]

    def _game_minutes_per_real_minute(self) -> float:
        """The actual, user-facing clock speed -- ticks_per_real_minute is
        just the internal beat (see the class docstring); what a person
        cares about is how many in-game minutes that beat adds up to.
        """
        return self._environment.minutes_per_tick * self._ticks_per_real_minute

    def _llm_calls_per_real_minute(self) -> float:
        """llm_calls_per_game_hour, converted to real time via how many
        game-hours actually pass per real minute:

            game-hr/real-min = game_minutes_per_real_minute / 60
            calls/real-min = llm_calls_per_game_hour * game-hr/real-min
        """
        game_hours_per_real_minute = self._game_minutes_per_real_minute() / 60.0
        return self._llm_calls_per_game_hour * game_hours_per_real_minute

    def state(self) -> dict[str, Any]:
        with self._traveler_lock:
            traveler_arrivals = list(self._traveler_arrivals)
        with self._speech_lock:
            speech = list(self._speech)
        return {
            "traveler_arrivals": traveler_arrivals,
            "speech": speech,
            "clock": self._environment.clock,
            "time_of_day": self._environment.time_of_day.value,
            "weather": self._environment.weather.value,
            "paused": self.is_paused(),
            "backend": self._backend,
            "game_minutes_per_real_minute": self._game_minutes_per_real_minute(),
            "llm_calls_per_game_hour": self._llm_calls_per_game_hour,
            "llm_calls_per_real_minute": self._llm_calls_per_real_minute(),
            "npcs": [
                {
                    "name": agent.identity.name,
                    "x": agent.position[0],
                    "y": agent.position[1],
                    "vx": agent.velocity[0],
                    "vy": agent.velocity[1],
                    "destination": None if agent.destination is None else list(agent.destination),
                    "busy": self._registry.is_busy(agent.identity.name),
                    "talking_to": self._registry.partner_of(agent.identity.name),
                    # A traveler stopped in its tracks while its turn is
                    # decided (see _run_movement_loop) -- the page has to
                    # know, or it keeps walking them on and snaps them back.
                    "deciding": agent.identity.name in self._turns_in_flight,
                    "traveler": self._registry.is_traveler(agent.identity.name),
                    "activity": self._last_activity.get(agent.identity.name, ""),
                }
                for agent in self._registry.all()
            ],
        }


@dataclass
class _TravelerState:
    exit_point: tuple[int, int]
    arrived_minute: int
    last_turn_minute: int
    # Everyone this traveler has already been prompted about -- each
    # person distracts them at most once.
    noticed: set[str] = field(default_factory=set)
    # Where it was last seen walking to, so arriving there can be noticed.
    heading_to: tuple[int, int] | None = None
    # Why it stopped in town, and whether it has been within
    # INTERACTION_RANGE of that purpose's place yet.
    purpose: TravelerPurpose = field(default_factory=lambda: TravelerPurpose("passing_through"))
    visited: bool = False
    # For a buyer: how many of the item it carried on arrival, so having
    # more means the purchase happened.
    item_baseline: int = 0
    # Someone it tried to start a conversation with who was busy, and since
    # when -- it gets a turn the moment they're free, or gives up and
    # leaves after TRAVELER_MAX_WAIT_MINUTES (see _update_travelers).
    waiting_for: str | None = None
    waiting_since: int = 0
    # Done here and heading out (gave up waiting, or stayed too long): no
    # more turns, just the walk to the exit point.
    leaving: bool = False
    # Set when a conversation it was in just ended (to the partner's name),
    # so it gets a turn to decide what's next right away.
    talked_with: str | None = None

    def reminder(self, agent: Agent, now: int) -> str:
        """Tacked onto a turn's stimulus, so the traveler knows where it
        stands on its purpose -- and how long it's been in town, which is
        what lets "stay a while" ever end.
        """
        bought = self.purpose.item is not None and agent.inventory.count(self.purpose.item) > self.item_baseline
        stayed = now - self.arrived_minute
        if stayed < 30:
            return self.purpose.reminder(arrived=self.visited, bought=bought)
        hours, minutes = divmod(stayed, 60)
        spent = f"{hours} hour{'s' if hours != 1 else ''}" + (f" {minutes} minutes" if minutes else "") if hours else f"{minutes} minutes"
        return self.purpose.reminder(arrived=self.visited, bought=bought) + f" You've been in town {spent} so far."


def _traveler_brief(sketch: Traveler, purpose: TravelerPurpose) -> str:
    # The purpose, not the sketch's own `reason`, says why they're here --
    # the two are rolled independently and could contradict each other.
    return (
        f"A traveler is arriving in town. They come from {sketch.origin}. {purpose.brief()} "
        f"At a glance they seem {' and '.join(sketch.traits)}."
    )


def _fallback_identity(sketch: Traveler, purpose: TravelerPurpose, *, money: int) -> Identity:
    """The environment agent's own sketch, filled out into an Identity
    without an LLM -- used whenever generation fails (see
    generate_traveler_identity).
    """
    return Identity(
        name=sketch.name,
        traits=list(sketch.traits),
        backstory=f"A traveler from {sketch.origin}. {purpose.brief()}",
        speech_style="plain and brief, like a stranger in town",
        goals=[purpose.goal()],
        habits=["Wanders through town without lingering long"],
        starting_money=money,
        gender=sketch.gender,
    )
