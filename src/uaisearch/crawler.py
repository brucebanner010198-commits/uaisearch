import json
import os
import time
import urllib.robotparser as robotparser
from collections import deque
from urllib.parse import urlparse

import httpx


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


class Frontier:
    def __init__(self, http_client: httpx.Client | None = None):
        self._http = http_client or httpx.Client(timeout=10.0)
        self._robots_cache: dict[str, robotparser.RobotFileParser] = {}
        self._last_fetch: dict[str, float] = {}

    def _get_robots(self, domain: str) -> robotparser.RobotFileParser:
        if domain not in self._robots_cache:
            rp = robotparser.RobotFileParser()
            try:
                resp = self._http.get(f"https://{domain}/robots.txt")
                rp.parse(resp.text.splitlines() if resp.status_code == 200 else [])
            except httpx.HTTPError:
                rp.parse([])  # unreachable robots.txt -> default allow
            self._robots_cache[domain] = rp
        return self._robots_cache[domain]

    def can_fetch(self, url: str, user_agent: str = "uaisearch-bot") -> bool:
        domain = urlparse(url).netloc
        return self._get_robots(domain).can_fetch(user_agent, url)

    def crawl_delay(self, domain: str, user_agent: str = "uaisearch-bot") -> float:
        delay = self._get_robots(domain).crawl_delay(user_agent)
        return float(delay) if delay else 1.0

    def wait_if_needed(self, domain: str) -> None:
        delay = self.crawl_delay(domain)
        elapsed = time.monotonic() - self._last_fetch.get(domain, 0.0)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_fetch[domain] = time.monotonic()


DARK_WEB_SUFFIXES = (".onion", ".i2p")


def is_dark_web(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host.lower().endswith(DARK_WEB_SUFFIXES)
