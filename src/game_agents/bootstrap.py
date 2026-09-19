"""Shared construction logic for every entry point that needs a live
NPCRegistry (the CLI in __main__.py, the mini-map server in
mini_map/game.py) -- kept in one place so both build the same NPCs, tools,
and data paths instead of drifting apart.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .chatty_mock import ChattyMockLLMClient
from .llm import AnthropicLLMClient, DeepSeekLLMClient, LLMClient
from .registry import NPCRegistry

DATA_DIR = Path(__file__).parent / "data"
NPCS_PATH = DATA_DIR / "npcs.json"
MEMORY_DIR = DATA_DIR / "memory"
INSTRUCTIONS_PATH = DATA_DIR / "instructions.json"
CONVERSATION_LOG_DIR = DATA_DIR / "conversation_log"
INVENTORY_DIR = DATA_DIR / "inventory"
WORLD_PATH = DATA_DIR / "world.json"
CONVERSATION_MAX_TURNS = 10

_BACKENDS: dict[str, type[LLMClient]] = {
    # The social demo mock, not llm.MockLLMClient (which only echoes) -- so
    # the mini-map has something to watch without an API key.
    "mock": ChattyMockLLMClient,
    "anthropic": AnthropicLLMClient,
    "deepseek": DeepSeekLLMClient,
}

load_dotenv(Path(__file__).parent / ".env")


def _build_llm() -> tuple[LLMClient, str]:
    name = os.environ.get("GAME_AGENTS_LLM", "mock")
    backend_cls = _BACKENDS.get(name)
    if backend_cls is None:
        raise ValueError(f"unknown GAME_AGENTS_LLM={name!r}, expected one of {sorted(_BACKENDS)}")
    model = os.environ.get("GAME_AGENTS_MODEL")
    kwargs = {"model": model} if model and name != "mock" else {}
    return backend_cls(**kwargs), name


def build_registry() -> tuple[NPCRegistry, str]:
    llm, backend = _build_llm()
    return (
        NPCRegistry(
            NPCS_PATH,
            MEMORY_DIR,
            llm,
            instructions_path=INSTRUCTIONS_PATH,
            # A backstop, not a script -- conversations normally end on a
            # goodbye well before this (see conversation.run_conversation).
            conversation_turns=CONVERSATION_MAX_TURNS,
            conversation_log_dir=CONVERSATION_LOG_DIR,
            inventory_dir=INVENTORY_DIR,
            world_path=WORLD_PATH,
        ),
        backend,
    )
