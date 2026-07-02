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
