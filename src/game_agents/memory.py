from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Memory:
    content: str
    importance: int  # 1-10, higher = more likely to surface later
    tags: set[str] = field(default_factory=set)
    timestamp: float = field(default_factory=time.time)


class MemoryStore:
    """A ranked memory stream, not a flat log. Retrieval narrows by tags (who
    or what a memory involves) then ranks the rest by recency and
    importance -- untagged memories are treated as general knowledge and
    always eligible, so "I'm hungry" can surface in any conversation while
    "Bob borrowed my hammer" only surfaces around Bob.
    """

    def __init__(self, *, recency_half_life_s: float = 3600.0) -> None:
        self._memories: list[Memory] = []
        self._recency_half_life_s = recency_half_life_s

    def add(
        self, content: str, *, importance: int, tags: set[str] | None = None, timestamp: float | None = None
    ) -> Memory:
        memory = Memory(content=content, importance=importance, tags=tags or set())
        if timestamp is not None:
            memory.timestamp = timestamp
        self._memories.append(memory)
        return memory

    def retrieve(self, *, tags: set[str] | None = None, limit: int = 8) -> list[Memory]:
        now = time.time()

        def relevant(m: Memory) -> bool:
            return not tags or not m.tags or bool(m.tags & tags)

        def score(m: Memory) -> float:
            recency = 0.5 ** ((now - m.timestamp) / self._recency_half_life_s)
            return recency * 0.6 + (m.importance / 10.0) * 0.4

        candidates = [m for m in self._memories if relevant(m)]
        return sorted(candidates, key=score, reverse=True)[:limit]

    def all(self) -> list[Memory]:
        return list(self._memories)
