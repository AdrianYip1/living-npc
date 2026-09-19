"""Manual smoke test for the agent core. Run with `python -m game_agents`
from src/ to talk to any NPC in data/npcs.json, loaded through the
registry so each NPC's memory loads from and saves back to data/memory/
independently.
"""
from __future__ import annotations

from .agent import Agent
from .bootstrap import build_registry
from .registry import NPCRegistry


def _choose_agent(registry: NPCRegistry) -> Agent:
    agents = registry.all()
    if len(agents) == 1:
        return agents[0]
    print("Who do you want to talk to?")
    for agent in agents:
        print(f"  {agent.identity.name}")
    while True:
        choice = input("> ").strip()
        agent = registry.get(choice)
        if agent is not None:
            return agent
        print("no NPC with that name, try again")


def main() -> None:
    registry, backend = build_registry()
    agent = _choose_agent(registry)
    print(f"Talking to {agent.identity.name} ({backend} backend, {len(agent.memory.all())} memories loaded). Ctrl+C to quit.")
    try:
        while True:
            try:
                text = input("you> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not text.strip():
                continue
            result = agent.respond(text, tags={"player"})
            if result.utterance is not None:
                print(f"{agent.identity.name}> {result.utterance}")
            else:
                print(f"{agent.identity.name}> *{result.action['name']}*")
    finally:
        registry.save_all()


if __name__ == "__main__":
    main()
