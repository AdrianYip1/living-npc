"""Shared construction logic for every entry point that needs a live
NPCRegistry (the CLI in __main__.py, the mini-map server in
mini_map/game.py) -- kept in one place so both build the same NPCs, tools,
and data paths instead of drifting apart.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

from .llm import AnthropicLLMClient, DeepSeekLLMClient, LLMClient, OpenAILLMClient
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
# What each resident knows the player as (see NPCRegistry).
PLAYER_NAMES_PATH = DATA_DIR / "player_names.json"
CONVERSATION_MAX_TURNS = 15

_BACKENDS: dict[str, type[LLMClient]] = {
    "anthropic": AnthropicLLMClient,
    "deepseek": DeepSeekLLMClient,
    "openai": OpenAILLMClient,
}


def _build_llm() -> tuple[LLMClient, str]:
    name = os.environ.get("GAME_AGENTS_LLM", "anthropic")
    backend_cls = _BACKENDS.get(name)
    if backend_cls is None:
        raise ValueError(f"unknown GAME_AGENTS_LLM={name!r}, expected one of {sorted(_BACKENDS)}")
    model = os.environ.get("GAME_AGENTS_MODEL")
    kwargs = {"model": model} if model else {}
    return backend_cls(**kwargs), name


# Everything a run leaves behind that the next one would otherwise pick up
# -- what the NPCs remember, what they're carrying, and what they call the
# player. Moved aside together by start_fresh(): a town that remembers you
# but has forgotten your name (or vice versa) is worse than either.
#
# Not TRAVELER_NAMES_PATH: it exists to stop the same traveler name coming
# round again, and that's as true of a fresh run as any other.
RUN_STATE_PATHS = (MEMORY_DIR, INVENTORY_DIR, PLAYER_NAMES_PATH)


def start_fresh(paths: tuple[Path, ...] = RUN_STATE_PATHS, *, data_dir: Path = DATA_DIR) -> Path | None:
    """Moves the last run's state into data/backup_<timestamp>/ so the next
    one starts with a town that's never met anyone. Returns where it went,
    or None if there was nothing to move.

    Moved, not deleted: a demo run is exactly when you find out you wanted
    yesterday's memories after all. Putting them back is a matter of moving
    the folders back. (Every loader treats a missing file as "nothing yet"
    -- see storage.load_memory -- so taking these away is enough.)
    """
    present = [path for path in paths if path.exists()]
    if not present:
        return None
    # Matches the backup_*/ already in game_agents/.gitignore.
    stamp = time.strftime("backup_%Y%m%d_%H%M%S")
    destination = data_dir / stamp
    suffix = 2
    while destination.exists():  # two fresh starts in the same second
        destination = data_dir / f"{stamp}_{suffix}"
        suffix += 1
    destination.mkdir(parents=True)
    for path in present:
        path.rename(destination / path.name)
    return destination


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
            player_names_path=PLAYER_NAMES_PATH,
        ),
        backend,
    )
