import json
import os
from collections import deque


class SeedManager:
    def __init__(self, seed_urls: list[str]):
        self._queue: deque[str] = deque(seed_urls)
        self._seen: set[str] = set(seed_urls)

    def next_url(self) -> str | None:
        return self._queue.popleft() if self._queue else None

    def add_discovered(self, url: str) -> None:
        if url not in self._seen:
            self._seen.add(url)
            self._queue.append(url)

    def __len__(self) -> int:
        return len(self._queue)

    def to_dict(self) -> dict:
        return {"queue": list(self._queue), "seen": list(self._seen)}

    @classmethod
    def from_dict(cls, data: dict) -> "SeedManager":
        manager = cls([])
        manager._queue = deque(data["queue"])
        manager._seen = set(data["seen"])
        return manager

    def save(self, path: str) -> None:
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.to_dict(), f)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: str, default_seeds: list[str]) -> "SeedManager":
        if os.path.exists(path):
            # ponytail: a corrupt frontier file must never wedge the scheduled crawler — fall back to seeds
            try:
                with open(path) as f:
                    return cls.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError, OSError):
                return cls(default_seeds)
        return cls(default_seeds)
