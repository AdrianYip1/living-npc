"""Manual smoke test for the agent core. Run with `python -m game_agents`
from src/ to talk to any NPC in data/npcs.json, loaded through the
registry so each NPC's memory loads from and saves back to data/memory/
independently.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .agent import Agent
from .llm import AnthropicLLMClient, DeepSeekLLMClient, LLMClient, MockLLMClient
from .registry import NPCRegistry
from .tools import Tool, ToolRegistry

DATA_DIR = Path(__file__).parent / "data"
NPCS_PATH = DATA_DIR / "npcs.json"
MEMORY_DIR = DATA_DIR / "memory"
INSTRUCTIONS_PATH = DATA_DIR / "instructions.json"
CONVERSATION_LOG_DIR = DATA_DIR / "conversation_log"

_BACKENDS: dict[str, type[LLMClient]] = {
    "mock": MockLLMClient,
    "anthropic": AnthropicLLMClient,
    "deepseek": DeepSeekLLMClient,
}

load_dotenv(Path(__file__).parent / ".env")


def _wave_handler(target: str) -> str:
    print(f"  [action] waves at {target}")
    return "waved"


def _build_llm() -> tuple[LLMClient, str]:
    name = os.environ.get("GAME_AGENTS_LLM", "mock")
    backend_cls = _BACKENDS.get(name)
    if backend_cls is None:
        raise ValueError(f"unknown GAME_AGENTS_LLM={name!r}, expected one of {sorted(_BACKENDS)}")
    model = os.environ.get("GAME_AGENTS_MODEL")
    kwargs = {"model": model} if model and name != "mock" else {}
    return backend_cls(**kwargs), name


def build_registry() -> tuple[NPCRegistry, str]:
    tools = ToolRegistry()
    tools.register(
        Tool(
            name="wave",
            description="Wave at someone",
            parameters={
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            handler=_wave_handler,
        )
    )
    llm, backend = _build_llm()
    return (
        NPCRegistry(
            NPCS_PATH,
            MEMORY_DIR,
            llm,
            tools,
            instructions_path=INSTRUCTIONS_PATH,
            conversation_log_dir=CONVERSATION_LOG_DIR,
        ),
        backend,
    )


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
