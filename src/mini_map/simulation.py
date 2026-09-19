"""Ties the environment agent's tick to the NPC registry -- the piece that
makes launching the mini-map "activate" the world instead of just drawing
it. Lives here (not in game_agents or environment_agent) so neither agent
package has to know the other exists; this is the only thing that imports
both.
"""
from __future__ import annotations

import threading
from typing import Any

from environment_agent.agent import EnvironmentAgent

from game_agents.agent import Scene, TurnResult
from game_agents.registry import NPCRegistry


class Simulation:
    """Every tick_interval_s seconds: the environment agent decides the new
    time of day (and weather, and whether a traveler arrives -- see its own
    docstring, it's meant to be the thing that re-triggers NPC idle
    behavior, not the other way around), then every NPC who isn't currently
    busy gets exactly one Agent.respond() call stimulated by that time of
    day. Matches the rest of the codebase's one-shot, no-lookahead turn
    shape -- this just triggers it autonomously instead of from a player.
    """

    def __init__(
        self,
        registry: NPCRegistry,
        environment: EnvironmentAgent,
        *,
        tick_interval_s: float = 15.0,
    ) -> None:
        self._registry = registry
        self._environment = environment
        self._tick_interval_s = tick_interval_s
        self._stop = threading.Event()
        # Read by the HTTP handler thread, written by the tick thread below;
        # plain dict/tuple assignment is fine to read concurrently under the
        # GIL for a display that's allowed to be a beat stale.
        self._last_activity: dict[str, str] = {}

    def run_forever(self) -> None:
        while not self._stop.is_set():
            self.tick()
            self._stop.wait(self._tick_interval_s)

    def stop(self) -> None:
        self._stop.set()

    def tick(self) -> None:
        event = self._environment.tick()
        phase = event.time_of_day.value

        # An NPC who was pulled into a conversation by someone else's turn
        # this tick has already acted -- ticking them again would be a
        # second turn in the same beat, not a fresh moment.
        acted: set[str] = set()
        for agent in self._registry.all():
            name = agent.identity.name
            if name in acted or self._registry.is_busy(name):
                continue

            scene = Scene(
                time=phase,
                context=f"It's {phase} and {event.weather.value} out. Decide what you do right now, guided by your habits.",
            )
            result = agent.respond(f"It is now {phase}.", scene=scene)
            acted.add(name)
            self._last_activity[name] = self._describe(result)

            if result.action is not None and result.action["name"] == "initiate_conversation":
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
            time=self._environment.time_of_day.value,
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

    def state(self) -> dict[str, Any]:
        return {
            "time_of_day": self._environment.time_of_day.value,
            "weather": self._environment.weather.value,
            "npcs": [
                {
                    "name": agent.identity.name,
                    "x": agent.position[0],
                    "y": agent.position[1],
                    "busy": self._registry.is_busy(agent.identity.name),
                    "activity": self._last_activity.get(agent.identity.name, ""),
                }
                for agent in self._registry.all()
            ],
        }
