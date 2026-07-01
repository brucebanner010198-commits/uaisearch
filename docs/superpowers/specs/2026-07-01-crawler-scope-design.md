# Crawler Scope: Deep-Web Discovery with Explicit Dark-Web Exclusion

## Problem

The Crawler plan (`docs/superpowers/plans/2026-07-01-crawler.md`) as originally written only fetches a fixed, hand-maintained seed list — it never discovers new URLs on its own. That's insufficient for the project's actual goal: reach **all legal content**, including long-tail "deep web" sites that no one has manually seeded, while never reaching the **dark web** (Tor `.onion`, I2P `.i2p`).

This spec covers three decisions needed to close that gap, and how they change the Crawler and Deployment plans.

## Decision 1: Discovery mechanism — cross-domain link-following

**Approaches considered:**
- **A. Cross-domain spidering (chosen).** The crawler extracts every outbound link from each fetched page and queues it, regardless of domain. This is the only approach that grows coverage toward "all legal content" over time. Trade-off: unbounded — without a scoring signal, the crawler will drift toward whatever's most link-dense (link farms, boilerplate-heavy sites) rather than valuable long-tail content, and may reach low-quality or borderline domains before the operator blocklist (Indexer plan, Task 8) reactively excludes them.
- **B. Common-Crawl-only, no live spidering.** Lean entirely on the existing Common Crawl ingestion path (Crawler plan Tasks 6/8) for breadth. Zero new engineering, but coverage is capped by whichever segments are chosen for backfill — it's a bigger backfill, not a self-growing crawler.
- **C. Bounded same-site spidering.** Follow links only within already-seeded domains. Safer, but never discovers new sites on its own — still requires manually seeding every domain of interest.

Given the explicit goal of reaching all legal content, Option A is the only one that matches. The existing operator blocklist (empty by default, Indexer plan) remains the safety valve for anything link-following pulls in that shouldn't be indexed.

## Decision 2: Dark-web exclusion — explicit rule, not incidental blocking

**Approaches considered:**
- **A. Explicit rule + no-proxy guarantee (chosen).** A tested `is_dark_web(url)` function rejects any URL whose host ends in `.onion` or `.i2p`, checked both before fetching a seed and before queuing a link discovered on a page. Additionally, no httpx client in this module is ever constructed with a `proxy=` argument pointed at Tor/I2P, so even a hypothetical bug in the rule can't result in a live fetch — plain DNS can't resolve those hosts without such a proxy. Two independent, testable layers.
- **B. Protocol allowlist only.** Only ever fetch `http`/`https` with standard DNS resolution, with no special-cased "dark web" concept. Behaviorally equivalent, but the exclusion is an implicit side effect of scoping rather than a documented, auditable rule — no artifact proves the exclusion was deliberate.
- **C. Rely on network-level inability to resolve `.onion`.** True today with no code changes, but accidental: if a proxy were ever added for an unrelated reason (e.g. corporate egress), the exclusion could silently stop holding, with no test to catch it.

Option A is chosen because this is a legal/policy requirement, not just a technical default — it needs a rule and a test that can be pointed to as proof, not an emergent property of how the fetcher happens to be written.

## Decision 3: Frontier persistence — local JSON file

**Approaches considered:**
- **A. Local JSON file on a Docker volume (chosen).** `SeedManager` gains `save()`/`load()`; the scheduled crawler run loads the queue + seen-set from `/data/frontier.json` at start and saves it at the end. A new named volume (`crawler-data`) keeps the file across container restarts. Simple, no new dependency, decoupled from the search index.
- **B. Store frontier state in OpenSearch.** Reuses the datastore already in the stack, but OpenSearch is a search index, not a work queue — using it for queue/seen-set state mixes concerns and is an unusual fit for its intended use.
- **C. No persistence.** Simplest, but discards every discovery at the end of each run, which walks back Decision 1 — spidering would never compound.

Option A is chosen: without it, the cross-domain spidering added in Decision 1 restarts from the fixed seed list every scheduled run and never actually grows.

## Architecture

```
scheduled run start
  → SeedManager.load(frontier.json, default_seeds=SEED_URLS)   [restore last run's discoveries]
  → run_crawl_cycle():
       for each url from seeds.next_url():
         if is_dark_web(url): skip                              [safety layer 1 — new]
         if not frontier.can_fetch(url): skip                    [existing robots.txt check]
         fetch → extract_clean_text → build_extracted_page
         extract_links(html, url) → filter dark-web + non-http → seeds.add_discovered(...)
  → SeedManager.save(frontier.json)                              [persist growth for next run — new]
  → indexer applies operator blocklist (existing, unchanged)     [safety layer 2, reactive]
```

Two of the three safety layers already existed in the Crawler plan (robots.txt respect, operator blocklist at index time). The new piece is the dark-web check at the front of the pipeline, and the frontier-persistence loop wrapping the whole cycle.

## Components

All four additions slot into existing Crawler-plan tasks — no task renumbering, so the Deployment plan's references to "Crawler plan Task 6 and 8" stay valid:

- **`SeedManager` (Task 1)** gains `to_dict()`/`from_dict()` (pure serialization of the queue and seen-set) and `save(path)`/`load(path, default_seeds)` (file I/O wrapping those). `load()` falls back to `default_seeds` when the file doesn't exist yet (first run ever).
- **`is_dark_web(url)` (Task 2, the Frontier task)** — a module-level function, not a `Frontier` method, since it's a pure URL check with no state: `urlparse(url).netloc.lower().endswith((".onion", ".i2p"))`.
- **`extract_links(html, base_url)` (Task 4, the Extractor task)** — walks `<a href>` tags via BeautifulSoup (already a dependency), resolves each against `base_url` with `urljoin` (stdlib), and drops anything that isn't `http`/`https` or fails `is_dark_web`.
- **`run_crawl_cycle` (Task 7)** — gets the `is_dark_web` skip check added at the top of its loop, and a call to `extract_links` + `seeds.add_discovered(...)` after each successful fetch.

On the Deployment side (Task 4, the scheduled crawler service): `scripts/run_crawler.py` loads and saves frontier state via a `FRONTIER_STATE_PATH` env var, and `docker-compose.yml`'s `crawler` service gains a `crawler-data:/data` volume mount.

## Error Handling

No new error-handling paths beyond what the existing plan already covers (`httpx.HTTPError` on fetch is already caught and skipped in `run_crawl_cycle`). `SeedManager.load()` handles the missing-file case (first run) by falling back to `default_seeds` rather than raising. `extract_links` silently drops malformed/non-http links rather than raising, since a single bad `href` on a page shouldn't abort processing the rest of that page's links.

## Testing

- `SeedManager`: round-trip `save()`/`load()` preserves queue and seen-set; `load()` falls back to `default_seeds` when the file doesn't exist.
- `is_dark_web`: `.onion` and `.i2p` hosts return `True`; an ordinary `.com` host returns `False`.
- `extract_links`: resolves a relative `href` against `base_url`; skips `mailto:`/`javascript:` links; skips a `.onion` link.
- `run_crawl_cycle`: a `.onion` URL in the seed list is skipped without a fetch attempt; links found in a fetched page's HTML end up queued in `seeds`.
- Manual verification (not automatable — needs a real network target): run one real crawl cycle against a permissive seed and confirm the `SeedManager` queue grows via discovered links, and that a synthetic `.onion` URL added to seeds never appears in the resulting pages.
