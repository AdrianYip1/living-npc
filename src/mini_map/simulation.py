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
from environment_agent.travelers import Traveler

from game_agents.agent import Agent, Scene, TurnResult
from game_agents.identity import Identity
from game_agents.registry import NPCRegistry, conversation_happened
from game_agents.storage import identity_to_record
from game_agents.traveler_identity import generate_traveler_identity
from game_agents.world import INTERACTION_RANGE, MAP_MAX, MAP_MIN, distance

log = logging.getLogger(__name__)


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
    opposite edge, and that it wants to visit one of world.json's places
    (picked at random) on the way.

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
    stops in its tracks, rather than walking on past whatever it noticed.
    It leaves town (is removed) on reaching its exit point.
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
    # exit point without asking -- a backstop for a model that never moves
    # (e.g. the mock backend, which only ever speaks), not a script.
    TRAVELER_MAX_STAY_MINUTES = 4 * 60
    # How close counts as having reached the exit point, in map units.
    TRAVELER_EXIT_REACHED = 1.0

    def __init__(
        self,
        registry: NPCRegistry,
        environment: EnvironmentAgent,
        *,
        ticks_per_real_minute: float = 60.0,
        llm_calls_per_game_hour: float = 4.0,
        seed: int | None = None,
    ) -> None:
        self._registry = registry
        self._environment = environment
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
        # Background work (identity generation, traveler turns) not yet
        # finished -- see wait_for_pending_travelers().
        self._pending: list[Future] = []
        self._pending_lock = threading.Lock()
        # Per-traveler bookkeeping, keyed by name, and which travelers
        # have a turn in flight right now (at most one each).
        self._traveler_state: dict[str, _TravelerState] = {}
        self._turns_in_flight: set[str] = set()
        self._turns_lock = threading.Lock()
        self._rng = random.Random(seed)

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

    def wait_for_pending_travelers(self, timeout: float = 10.0) -> None:
        """Blocks until all traveler background work is done -- identity
        generation, plus any turns that queued (e.g. the arrival turn each
        admitted traveler gets). For tests and other callers that need
        arrivals to have fully landed before they look.
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
                self._registry.step_movement(dt, holding=set(self._turns_in_flight))
                self._update_travelers()
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
        identity = generate_traveler_identity(
            self._registry.llm,
            _traveler_brief(sketch),
            fallback=_fallback_identity(sketch),
            taken_names=[agent.identity.name for agent in self._registry.all()] + recent_names,
        )
        # The fallback skips the taken-name check, and two generations can
        # race to the same name -- the registry has the final say.
        identity.name = self._registry.unique_name(identity.name)

        entry, exit_point = self._pick_entry_and_exit()
        # One of world.json's places, picked at random -- a reason to stop
        # somewhere in town rather than just crossing it.
        places = self._registry.places
        visit = self._rng.choice(places) if places else None
        standing_context = (
            f"Your exit point: ({exit_point[0]}, {exit_point[1]}), on the edge of the map. "
            "Reaching it means leaving town."
        )
        if visit is not None:
            standing_context += f" Before you leave, you want to pay a visit to {_describe_place(visit)}."
        self._registry.add_traveler(identity, position=entry, standing_context=standing_context)
        now = self._environment.elapsed_minutes
        self._traveler_state[identity.name] = _TravelerState(
            exit_point=exit_point, arrived_minute=now, last_turn_minute=now, visit=visit
        )

        # The id is assigned only now, once the traveler exists, so ids in
        # state() always appear in increasing order even if two
        # generations finish out of order.
        with self._traveler_lock:
            self._traveler_arrivals.append(
                {"id": self._next_traveler_id, "arrived_at": clock, "identity": identity_to_record(identity)}
            )
            self._next_traveler_id += 1
            del self._traveler_arrivals[: -self.RECENT_TRAVELERS]

        self._start_traveler_turn(
            identity.name,
            f"You've just arrived at the edge of town, at ({entry[0]}, {entry[1]}). "
            f"Your exit point is ({exit_point[0]}, {exit_point[1]}), on the far side of the map."
            + self._traveler_state[identity.name].visit_reminder(),
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

            if (
                st.visit is not None
                and not st.visited
                and distance(agent.position, st.visit["position"]) <= INTERACTION_RANGE
            ):
                st.visited = True

            if now - st.arrived_minute >= self.TRAVELER_MAX_STAY_MINUTES:
                if agent.destination != st.exit_point:
                    agent.destination = st.exit_point
                    self._last_activity[name] = "has lingered long enough and heads out of town"
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
                    f"Your exit point is ({st.exit_point[0]}, {st.exit_point[1]})." + st.visit_reminder(),
                )
                continue

            idle = agent.destination is None and agent.velocity == (0.0, 0.0)
            if idle and now - st.last_turn_minute >= self.TRAVELER_IDLE_TURN_MINUTES:
                self._start_traveler_turn(
                    name,
                    f"You're standing at ({agent.position[0]:.0f}, {agent.position[1]:.0f}), not walking anywhere. "
                    f"Your exit point is ({st.exit_point[0]}, {st.exit_point[1]})." + st.visit_reminder(),
                )

    def _start_traveler_turn(self, name: str, stimulus: str) -> None:
        with self._turns_lock:
            if name in self._turns_in_flight:
                return
            self._turns_in_flight.add(name)
        if not self._submit(self._turn_pool, self._run_traveler_turn, name, stimulus):
            with self._turns_lock:
                self._turns_in_flight.discard(name)

    def _run_traveler_turn(self, name: str, stimulus: str) -> None:
        try:
            agent = self._registry.get(name)
            if agent is None:
                return
            result = agent.respond(stimulus, scene=self._scene(), routine=True)
            self._record_turn(agent, result)
        except Exception:
            log.exception("traveler %s's turn failed", name)
        finally:
            st = self._traveler_state.get(name)
            if st is not None:
                st.last_turn_minute = self._environment.elapsed_minutes
            with self._turns_lock:
                self._turns_in_flight.discard(name)

    def _scene(self) -> Scene:
        clock, phase = self._environment.clock, self._environment.time_of_day.value
        return Scene(
            time=f"{clock} ({phase})",
            context=f"It's {clock}, {phase}, and {self._environment.weather.value} out.",
        )

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

            scene = Scene(
                time=f"{clock} ({phase})",
                context=f"It's {clock}, {phase}, and {weather} out. Decide what you do right now, guided by your habits.",
            )
            result = agent.respond(f"It is now {clock} ({phase}).", scene=scene, routine=True)
            acted.add(name)
            partner = self._record_turn(agent, result)
            if partner is not None:
                acted.add(partner)

    def _record_turn(self, agent: Agent, result: TurnResult) -> str | None:
        """Updates the activity line for whoever took a turn -- and, if the
        turn was a conversation that actually happened, for its partner
        too. Returns that partner's name, if any. A refused conversation
        (out of range, busy) leaves the target untouched.
        """
        name = agent.identity.name
        self._last_activity[name] = self._describe(result)
        if (
            result.action is not None
            and result.action["name"] == "initiate_conversation"
            and conversation_happened(result.action["result"])
        ):
            resolved = self._registry.resolve(result.action["arguments"].get("target_name", ""))
            if resolved is not None:
                self._last_activity[resolved] = f"talked with {name}"
                return resolved
        return None

    # ------------------------------------------------------------------ #
    # player conversation (see mini_map.game's /api/conversation/* routes)
    # ------------------------------------------------------------------ #
    def start_conversation(self, name: str) -> bool:
        """Claims the NPC as busy for the duration of a player conversation
        -- same busy set the world tick and NPC-to-NPC conversations check,
        so this NPC is skipped by both for as long as the player's talking
        to them, exactly as if they were mid-exchange with someone else.
        """
        if self._registry.get(name) is None:
            return False
        return self._registry.try_occupy(name)

    def end_conversation(self, name: str) -> None:
        self._registry.release(name)

    def say(self, name: str, text: str) -> dict[str, Any] | None:
        agent = self._registry.get(name)
        if agent is None:
            return None

        scene = Scene(
            time=f"{self._environment.clock} ({self._environment.time_of_day.value})",
            context=(
                f"The player has walked up and is speaking with you directly. "
                f"It's {self._environment.weather.value} out."
            ),
        )
        result = agent.respond(text, scene=scene, tags={"player"})
        self._last_activity[name] = self._describe(result)
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
        return {
            "traveler_arrivals": traveler_arrivals,
            "clock": self._environment.clock,
            "time_of_day": self._environment.time_of_day.value,
            "weather": self._environment.weather.value,
            "paused": self.is_paused(),
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
    # The world.json place it came to visit (None if the world has none),
    # and whether it has been within INTERACTION_RANGE of it yet.
    visit: dict[str, Any] | None = None
    visited: bool = False

    def visit_reminder(self) -> str:
        """Tacked onto a turn's stimulus, so the traveler knows whether its
        visit is still ahead of it or already done.
        """
        if self.visit is None:
            return ""
        if self.visited:
            return f" You've already visited {self.visit['name']}."
        return f" You still want to visit {_describe_place(self.visit)}."


def _describe_place(place: dict[str, Any]) -> str:
    return f"{place['name']} at ({place['position'][0]}, {place['position'][1]})"


def _traveler_brief(sketch: Traveler) -> str:
    return (
        f"A traveler is arriving in town. They come from {sketch.origin}, and they're {sketch.reason}. "
        f"At a glance they seem {' and '.join(sketch.traits)}."
    )


def _fallback_identity(sketch: Traveler) -> Identity:
    """The environment agent's own sketch, filled out into an Identity
    without an LLM -- used by the mock backend and whenever generation
    fails (see generate_traveler_identity).
    """
    return Identity(
        name=sketch.name,
        traits=list(sketch.traits),
        backstory=f"A traveler from {sketch.origin}, {sketch.reason}.",
        speech_style="plain and brief, like a stranger in town",
        goals=[sketch.reason],
        habits=["Wanders through town without lingering long"],
    )
