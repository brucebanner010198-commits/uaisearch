import json
import logging
import os
import re
import time
import urllib.robotparser as robotparser
from collections import deque
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from simhash import Simhash

from uaisearch.models import ExtractedPage

logger = logging.getLogger(__name__)


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


async def fetch(url: str, client: httpx.AsyncClient) -> str:
    response = await client.get(url, headers={"User-Agent": "uaisearch-bot/0.1"})
    response.raise_for_status()
    return response.text


AD_PATTERNS = re.compile(r"(doubleclick|googlesyndication|sponsor(ed)?|advert)", re.I)


def estimate_ad_density(soup: BeautifulSoup) -> float:
    all_tags = soup.find_all(True)
    if not all_tags:
        return 0.0
    ad_tags = [
        t for t in all_tags
        if AD_PATTERNS.search(" ".join(t.get("class", []))) or AD_PATTERNS.search(t.get("id", ""))
    ]
    return round(len(ad_tags) / len(all_tags), 4)


def strip_ad_elements(html: str) -> tuple[BeautifulSoup, float]:
    soup = BeautifulSoup(html, "lxml")
    ad_ratio = estimate_ad_density(soup)
    for tag in soup.find_all(True):
        if tag.decomposed:  # ancestor already stripped this subtree
            continue
        classes = " ".join(tag.get("class", []))
        tag_id = tag.get("id", "")
        src = tag.get("src", "") if tag.name == "iframe" else ""
        if AD_PATTERNS.search(classes) or AD_PATTERNS.search(tag_id) or AD_PATTERNS.search(src):
            tag.decompose()
    return soup, ad_ratio


def extract_clean_text(html: str, url: str) -> tuple[str, str, float]:
    soup, ad_ratio = strip_ad_elements(html)
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    body = trafilatura.extract(str(soup), url=url) or ""
    return title, body, ad_ratio


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for tag in soup.find_all("a", href=True):
        href = urljoin(base_url, tag["href"])
        if urlparse(href).scheme not in ("http", "https") or is_dark_web(href):
            continue
        links.append(href)
    return links


def build_extracted_page(html: str, url: str, domain: str, crawl_date: str) -> ExtractedPage:
    title, body, ad_ratio = extract_clean_text(html, url)
    return ExtractedPage(
        url=url, domain=domain, title=title, text=body,
        ad_ratio=ad_ratio, crawl_date=crawl_date, simhash=Simhash(body).value,
    )


from datetime import date


async def run_crawl_cycle(
    seeds: SeedManager,
    frontier: Frontier,
    client: httpx.AsyncClient,
    max_pages: int,
) -> list[ExtractedPage]:
    pages: list[ExtractedPage] = []
    while len(pages) < max_pages:
        url = seeds.next_url()
        if url is None:
            break
        if is_dark_web(url) or not frontier.can_fetch(url):
            continue
        domain = urlparse(url).netloc
        # ponytail: one bad page must not abort the whole crawl cycle — isolate per-URL failures
        try:
            frontier.wait_if_needed(domain)
            html = await fetch(url, client)
            pages.append(build_extracted_page(html, url, domain, date.today().isoformat()))
            for link in extract_links(html, url):
                seeds.add_discovered(link)
        except Exception as exc:  # untrusted markup — any page can fail extraction/fetch
            logger.warning("skipping %s: %s", url, exc)
            continue
    return pages
