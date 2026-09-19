"""Shared construction logic for every entry point that needs a live
NPCRegistry (the CLI in __main__.py, the mini-map server in
mini_map/game.py) -- kept in one place so both build the same NPCs, tools,
and data paths instead of drifting apart.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .llm import AnthropicLLMClient, DeepSeekLLMClient, LLMClient
from .registry import NPCRegistry

load_dotenv(Path(__file__).parent / ".env")

DATA_DIR = Path(__file__).parent / "data"
NPCS_PATH = DATA_DIR / "npcs.json"
MEMORY_DIR = DATA_DIR / "memory"
INSTRUCTIONS_PATH = DATA_DIR / "instructions.json"
# Also where the speech/animation program looks for the conversation feed
# (see conversation_export.py) -- overridable so it can point elsewhere.
CONVERSATION_LOG_DIR = Path(os.environ.get("CONVERSATION_LOG_DIR", DATA_DIR / "conversation_log"))
INVENTORY_DIR = DATA_DIR / "inventory"
WORLD_PATH = DATA_DIR / "world.json"
# Every name a traveler has gone by, across runs (see traveler_names.py).
TRAVELER_NAMES_PATH = DATA_DIR / "traveler_names.json"
CONVERSATION_MAX_TURNS = 15

_BACKENDS: dict[str, type[LLMClient]] = {
    "anthropic": AnthropicLLMClient,
    "deepseek": DeepSeekLLMClient,
}


def _build_llm() -> tuple[LLMClient, str]:
    name = os.environ.get("GAME_AGENTS_LLM", "anthropic")
    backend_cls = _BACKENDS.get(name)
    if backend_cls is None:
        raise ValueError(f"unknown GAME_AGENTS_LLM={name!r}, expected one of {sorted(_BACKENDS)}")
    model = os.environ.get("GAME_AGENTS_MODEL")
    kwargs = {"model": model} if model else {}
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
