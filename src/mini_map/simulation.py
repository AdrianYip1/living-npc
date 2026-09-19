"""Ties the environment agent's tick to the NPC registry -- the piece that
makes launching the mini-map "activate" the world instead of just drawing
it. Lives here (not in game_agents or environment_agent) so neither agent
package has to know the other exists; this is the only thing that imports
both.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from environment_agent.agent import EnvironmentAgent

from game_agents.agent import Scene, TurnResult
from game_agents.registry import NPCRegistry, conversation_happened


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
    """

    MOVEMENT_HZ = 30.0
    # How many recent traveler arrivals state() keeps around. The page
    # spawns each one once, by id, so this only needs to cover arrivals
    # between two polls -- the rest is slack for a slow/backgrounded tab.
    RECENT_TRAVELERS = 20
    # Cap on one movement step, so a stalled thread (or a long pause
    # resuming) doesn't fling NPCs across the map in a single jump.
    MAX_MOVEMENT_DT_S = 0.1

    def __init__(
        self,
        registry: NPCRegistry,
        environment: EnvironmentAgent,
        *,
        ticks_per_real_minute: float = 60.0,
        llm_calls_per_game_hour: float = 4.0,
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
        # already spawned. Appended on the environment thread, copied out
        # by state() on the HTTP thread -- hence the lock.
        self._traveler_arrivals: list[dict[str, Any]] = []
        self._next_traveler_id = 1
        self._traveler_lock = threading.Lock()

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
                self._registry.step_movement(dt)
            self._stop.wait(1.0 / self.MOVEMENT_HZ)

    def tick(self) -> None:
        """Force one full beat right now -- advance the environment and
        query every idle NPC, regardless of npc_query_interval_s. Not used
        by run_forever() (which paces the two separately); this is for
        tests and any other caller that wants an immediate, deterministic
        beat instead of waiting on the real-time schedule.
        """
        self._advance_environment()
        self._query_npcs()

    def _advance_environment(self) -> None:
        event = self._environment.tick()
        if not event.travelers_arrived:
            return
        with self._traveler_lock:
            for traveler in event.travelers_arrived:
                self._traveler_arrivals.append(
                    {
                        "id": self._next_traveler_id,
                        "name": traveler.name,
                        "origin": traveler.origin,
                        "reason": traveler.reason,
                        "traits": list(traveler.traits),
                        "arrived_at": event.clock,
                    }
                )
                self._next_traveler_id += 1
            del self._traveler_arrivals[: -self.RECENT_TRAVELERS]

    def _query_npcs(self) -> None:
        phase = self._environment.time_of_day.value
        weather = self._environment.weather.value
        clock = self._environment.clock

        # An NPC who was pulled into a conversation by someone else's turn
        # this round has already acted -- querying them again would be a
        # second turn in the same beat, not a fresh moment.
        acted: set[str] = set()
        for agent in self._registry.all():
            name = agent.identity.name
            if name in acted or self._registry.is_busy(name):
                continue

            scene = Scene(
                time=f"{clock} ({phase})",
                context=f"It's {clock}, {phase}, and {weather} out. Decide what you do right now, guided by your habits.",
            )
            result = agent.respond(f"It is now {clock} ({phase}).", scene=scene)
            acted.add(name)
            self._last_activity[name] = self._describe(result)

            # Only a conversation that actually happened pulls the target in --
            # a refused one (out of range, busy) leaves them free to act.
            if (
                result.action is not None
                and result.action["name"] == "initiate_conversation"
                and conversation_happened(result.action["result"])
            ):
                target_name = result.action["arguments"].get("target_name", "")
                resolved = self._registry.resolve(target_name)
                if resolved is not None:
                    acted.add(resolved)
                    self._last_activity[resolved] = f"talked with {name}"

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
                    "activity": self._last_activity.get(agent.identity.name, ""),
                }
                for agent in self._registry.all()
            ],
        }
