# Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire OpenSearch, the FastAPI app, the scheduled crawler, an optional self-hosted LLM, and TLS termination together into a Docker Compose stack that brings the whole system up on one VM.

**Architecture:** Four core services (`opensearch`, `app`, `crawler`, `caddy`) plus one opt-in profile-gated service (`local-llm`, Ollama). Bring-up order: OpenSearch → one-time index creation and Common Crawl backfill → app → scheduled crawler → Caddy in front for TLS. This plan does not introduce new application code beyond small bring-up/backfill scripts — it wires together functions already built in the Indexer, Crawler, Retrieval, Synthesis, and API plans.

**Tech Stack:** Docker, Docker Compose, Caddy (reverse proxy + automatic TLS), optionally Ollama (self-hosted LLM backend).

## Global Constraints

- Depends on the Indexer, Crawler, Retrieval, Synthesis, API, and Conversational UI plans being implemented first.
- `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL` must remain generic OpenAI-compatible config values — never hardcode a specific vendor's SDK or endpoint shape into compose or scripts (per the Synthesis plan's model-agnostic requirement).
- These verification steps use real `docker`/`docker compose` commands rather than `pytest` — infrastructure wiring isn't unit-testable the way application code is, but every step still has a concrete, checkable expected outcome.

---

### Task 1: Dockerfile for the app

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `scripts/.gitkeep`

**Interfaces:**
- Consumes: `pyproject.toml`, `src/uaisearch/` (all prior plans).
- Produces: a `uaisearch-app` image runnable with `uvicorn uaisearch.api:app`. Used by Task 2's `app` service.

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/

RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "uaisearch.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```
# .dockerignore
.venv/
__pycache__/
*.pyc
.git/
tests/
```

Create an empty `scripts/.gitkeep` so `COPY scripts/ scripts/` has something to copy — Task 3 fills this directory with real scripts and rebuilds the image.

- [ ] **Step 2: Build and verify**

Run: `docker build -t uaisearch-app .`
Expected: build succeeds.

Run: `docker run --rm uaisearch-app python -c "import uaisearch.api; print('ok')"`
Expected: prints `ok` (confirms the image has all dependencies needed to import the app; this doesn't require live OpenSearch/LLM since FastAPI dependencies resolve per-request, not at import time).

- [ ] **Step 3: Commit**

```bash
git add Dockerfile .dockerignore scripts/.gitkeep
git commit -m "feat: add Dockerfile for the app image"
```

---

### Task 2: Core Compose stack — OpenSearch and the app

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `.gitignore`

**Interfaces:**
- Consumes: `uaisearch-app` image (Task 1).
- Produces: `opensearch` and `app` services reachable at `localhost:9200` and `localhost:8000`. Extended by Tasks 4-6.

- [ ] **Step 1: Write the compose file and env template**

```yaml
# docker-compose.yml
services:
  opensearch:
    image: opensearchproject/opensearch:2.15.0
    environment:
      - discovery.type=single-node
      - DISABLE_SECURITY_PLUGIN=true
      - OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m
    ports:
      - "127.0.0.1:9200:9200"  # security plugin disabled — must never be reachable off-host
    volumes:
      - opensearch-data:/usr/share/opensearch/data
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:9200"]
      interval: 10s
      retries: 30
    restart: unless-stopped

  app:
    build: .
    depends_on:
      opensearch:
        condition: service_healthy
    environment:
      - OPENSEARCH_HOST=opensearch
      - OPENSEARCH_PORT=9200
      - TARGET_DOMAINS=${TARGET_DOMAINS}
      - BLOCKED_DOMAINS=${BLOCKED_DOMAINS}
      - LLM_BASE_URL=${LLM_BASE_URL}
      - LLM_API_KEY=${LLM_API_KEY}
      - LLM_MODEL=${LLM_MODEL}
    ports:
      - "127.0.0.1:8000:8000"  # public traffic goes through caddy so TLS and X-Forwarded-For rate limiting can't be bypassed
    restart: unless-stopped

volumes:
  opensearch-data:
```

```bash
# .env.example
DOMAIN=your-real-domain.example
LLM_BASE_URL=https://api.anthropic.com/v1
LLM_API_KEY=changeme
LLM_MODEL=claude-sonnet-5
SEED_URLS=https://example-niche-blog.com/
TARGET_DOMAINS=example-niche-blog.com
BLOCKED_DOMAINS=
```

`BLOCKED_DOMAINS` is the operator-maintained legal exclusion list (`is_blocked`, Indexer plan Task 8) — comma-separated domains or exact URLs to exclude, e.g. after a DMCA takedown notice. Empty by default.

Create `.gitignore`:

```
.env
__pycache__/
*.egg-info/
.pytest_cache/
```

Copy `.env.example` to `.env` and fill in real values before bring-up: `cp .env.example .env`.

- [ ] **Step 2: Bring up and verify**

Run: `docker compose up -d opensearch app`
Expected: both containers report `running` in `docker compose ps`.

Run: `curl -s http://localhost:9200 | python3 -m json.tool`
Expected: JSON banner with OpenSearch version info.

Run: `curl -s http://localhost:8000/openapi.json | python3 -m json.tool`
Expected: valid OpenAPI JSON (the app started and can serve requests, even before the index has any data).

Run: From any other machine, `curl http://<host>:9200` fails (refused/timeout).
Expected: OpenSearch is not accessible from off-host.

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml .env.example .gitignore
git commit -m "feat: add core Docker Compose stack (OpenSearch + app)"
```

---

### Task 3: Operational scripts — index creation, Common Crawl backfill, blocklist purge

**Files:**
- Create: `scripts/create_index.py`
- Create: `scripts/backfill_common_crawl.py`
- Create: `scripts/purge_blocked.py`

**Interfaces:**
- Consumes: `create_index`, `index_page`, `load_simhash_index` (`uaisearch.indexer`, Indexer plan), `get_client` (`uaisearch.opensearch_client`), `iter_wet_records`, `build_page_from_wet` (`uaisearch.common_crawl`, Crawler plan Task 6 and 8), `purge_blocked` (`uaisearch.indexer`, Indexer plan Task 9).
- Produces: three standalone scripts run manually during initial bring-up or as operator maintenance tasks.

- [ ] **Step 1: Write the scripts**

```python
# scripts/create_index.py
import os

from uaisearch.indexer import create_index
from uaisearch.opensearch_client import get_client


def main() -> None:
    client = get_client(
        host=os.environ.get("OPENSEARCH_HOST", "localhost"),
        port=int(os.environ.get("OPENSEARCH_PORT", "9200")),
    )
    create_index(client)
    print("Index ready.")


if __name__ == "__main__":
    main()
```

```python
# scripts/backfill_common_crawl.py
import os
import sys
from datetime import date

from uaisearch.common_crawl import build_page_from_wet, iter_wet_records
from uaisearch.indexer import create_index, index_page, load_simhash_index
from uaisearch.opensearch_client import get_client

TARGET_DOMAINS = {d for d in os.environ.get("TARGET_DOMAINS", "").split(",") if d}
BLOCKED_DOMAINS = {d for d in os.environ.get("BLOCKED_DOMAINS", "").split(",") if d}


def main(s3_key: str, crawl_date: str) -> None:
    client = get_client(
        host=os.environ.get("OPENSEARCH_HOST", "localhost"),
        port=int(os.environ.get("OPENSEARCH_PORT", "9200")),
    )
    create_index(client)
    dedup_index = load_simhash_index(client)

    total_chunks = 0
    for url, domain, text in iter_wet_records(s3_key, TARGET_DOMAINS):
        page = build_page_from_wet(url, domain, text, crawl_date=crawl_date)
        total_chunks += index_page(client, page, dedup_index, blocklist=BLOCKED_DOMAINS)
    print(f"Backfilled {total_chunks} chunks from {s3_key}.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: backfill_common_crawl.py <wet-segment-s3-key> [crawl-date YYYY-MM-DD]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat())
```

```python
# scripts/purge_blocked.py
import os

from uaisearch.indexer import purge_blocked
from uaisearch.opensearch_client import get_client

BLOCKED_DOMAINS = {d for d in os.environ.get("BLOCKED_DOMAINS", "").split(",") if d}


def main() -> None:
    client = get_client(
        host=os.environ.get("OPENSEARCH_HOST", "localhost"),
        port=int(os.environ.get("OPENSEARCH_PORT", "9200")),
    )
    deleted = purge_blocked(client, BLOCKED_DOMAINS)
    print(f"Purged {deleted} chunks for {len(BLOCKED_DOMAINS)} blocked entries.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rebuild the app image and verify**

The `app` image built in Task 1 predates these scripts — rebuild before using them:

Run: `docker compose build app && docker compose up -d app`

Run: `docker compose exec app python scripts/create_index.py`
Expected: prints `Index ready.`; `curl -s http://localhost:9200/_cat/indices` shows a `pages` index.

Run (with a real Common Crawl WET segment key, e.g. from a monthly crawl's `wet.paths.gz` listing, and `TARGET_DOMAINS` set in `.env` to your seed domains): `docker compose exec app python scripts/backfill_common_crawl.py "crawl-data/CC-MAIN-.../wet/....warc.wet.gz" YYYY-MM-DD`
Expected: prints `Backfilled N chunks from ...` with N > 0 if any target domains appear in that segment.

Note: env vars now reach `docker compose exec app` because they're in the service `environment:` block (Task 2).

Run: `docker compose exec app python scripts/purge_blocked.py`
Expected: `Purged 0 chunks for 0 blocked entries.` with the default empty list.

**Operator runbook:** To take content down, add the domain/URL to `BLOCKED_DOMAINS` in `.env`, run `docker compose up -d app crawler`, then `docker compose exec app python scripts/purge_blocked.py`.

- [ ] **Step 3: Commit**

```bash
git add scripts/create_index.py scripts/backfill_common_crawl.py scripts/purge_blocked.py
git commit -m "feat: add index creation, backfill, and blocklist purge scripts"
```

---

### Task 4: Scheduled crawler service

**Files:**
- Create: `scripts/run_crawler.py`
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: `SeedManager`, `Frontier`, `run_crawl_cycle` (`uaisearch.crawler`, Crawler plan), `index_page`, `load_simhash_index` (`uaisearch.indexer`, Indexer plan).
- Produces: a `crawler` compose service that periodically fetches seed URLs, follows discovered links, and indexes results, persisting its frontier across runs on a volume.

- [ ] **Step 1: Write the script and compose service**

```python
# scripts/run_crawler.py
import asyncio
import os

import httpx

from uaisearch.crawler import Frontier, SeedManager, run_crawl_cycle
from uaisearch.indexer import create_index, index_page, load_simhash_index
from uaisearch.opensearch_client import get_client

SEED_URLS = [u for u in os.environ.get("SEED_URLS", "").split(",") if u]
BLOCKED_DOMAINS = {d for d in os.environ.get("BLOCKED_DOMAINS", "").split(",") if d}
FRONTIER_STATE_PATH = os.environ.get("FRONTIER_STATE_PATH", "/data/frontier.json")


async def main() -> None:
    client = get_client(
        host=os.environ.get("OPENSEARCH_HOST", "localhost"),
        port=int(os.environ.get("OPENSEARCH_PORT", "9200")),
    )
    create_index(client)
    dedup_index = load_simhash_index(client)

    seeds = SeedManager.load(FRONTIER_STATE_PATH, default_seeds=SEED_URLS)
    frontier = Frontier()
    async with httpx.AsyncClient() as http_client:
        pages = await run_crawl_cycle(
            seeds, frontier, http_client,
            max_pages=int(os.environ.get("MAX_PAGES", "50")),
        )

    total_chunks = sum(
        index_page(client, page, dedup_index, blocklist=BLOCKED_DOMAINS) for page in pages
    )
    print(f"Crawled {len(pages)} pages, indexed {total_chunks} chunks.")

    # Save after indexing — a crash mid-index re-fetches this cycle instead of losing those pages forever
    seeds.save(FRONTIER_STATE_PATH)


if __name__ == "__main__":
    asyncio.run(main())
```

```yaml
# add to docker-compose.yml, under services:
  crawler:
    build: .
    depends_on:
      opensearch:
        condition: service_healthy
    environment:
      - OPENSEARCH_HOST=opensearch
      - OPENSEARCH_PORT=9200
      - SEED_URLS=${SEED_URLS}
      - BLOCKED_DOMAINS=${BLOCKED_DOMAINS}
      - MAX_PAGES=200
      - FRONTIER_STATE_PATH=/data/frontier.json
    volumes:
      - crawler-data:/data
    entrypoint: ["sh", "-c", "while true; do if python scripts/run_crawler.py; then sleep 86400; else echo 'crawler run failed; retrying in 15m'; sleep 900; fi; done"]
    restart: unless-stopped
```

```yaml
# add crawler-data to the existing volumes: block in docker-compose.yml
  crawler-data:
```

The `crawler-data` volume is what makes link-discovery compound across scheduled runs — without it, `FRONTIER_STATE_PATH` would never exist on container restart and every run would fall back to `SEED_URLS` alone (see the Crawler plan's `SeedManager.load()`, Task 1).

- [ ] **Step 2: Run and verify**

Run: `docker compose up -d crawler`
Expected: `docker compose logs crawler` shows `Crawled N pages, indexed M chunks.` within a few minutes (depends on `SEED_URLS` being reachable and permissive `robots.txt`).

Run: `curl -s "http://localhost:9200/pages/_count" | python3 -m json.tool`
Expected: `count` increased compared to before the crawler ran.

Run (after the container has completed at least one cycle): `docker compose exec crawler cat /data/frontier.json | python3 -m json.tool`
Expected: valid JSON with `queue` and `seen` keys, `seen` containing more URLs than `SEED_URLS` originally listed (confirms discovered links were persisted).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_crawler.py docker-compose.yml
git commit -m "feat: add scheduled crawler service with persisted frontier state"
```

---

### Task 5: Optional self-hosted LLM service

**Files:**
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: none.
- Produces: an opt-in `local-llm` service (Ollama, serves an OpenAI-compatible endpoint at `/v1`), only started when the `local-llm` Compose profile is enabled — the app's `LLM_BASE_URL`/`LLM_MODEL` env vars are what actually point at it, no code changes needed since `LLMClient` (Synthesis plan) is already backend-agnostic.

- [ ] **Step 1: Write the compose service**

```yaml
# add to docker-compose.yml, under services:
  local-llm:
    image: ollama/ollama:0.9.6
    profiles: ["local-llm"]
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    restart: unless-stopped
```

**Note:** Pin to a fixed release tag, never `latest` — bump to the current release at implementation time (github.com/ollama/ollama/releases).

```yaml
# add ollama-data to the existing volumes: block in docker-compose.yml
  ollama-data:
```

To use it, set in `.env`: `LLM_BASE_URL=http://local-llm:11434/v1`, `LLM_MODEL=<a standard, safety-tuned instruct model pulled into Ollama>`, and leave `LLM_API_KEY` empty (Ollama doesn't require one).

- [ ] **Step 2: Run and verify**

Run: `docker compose --profile local-llm up -d local-llm`
Expected: container starts; `curl -s http://localhost:11434` returns `Ollama is running`.

Run (after pulling a model into the running container, e.g. `docker compose exec local-llm ollama pull <model>`): restart the `app` service with `.env` pointed at `local-llm`, then exercise `/api/v1/answer` and confirm it returns a real generated answer instead of an error.

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: add optional self-hosted LLM service"
```

---

### Task 6: TLS termination

**Files:**
- Create: `Caddyfile`
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: `app` service (Task 2).
- Produces: a `caddy` service terminating TLS in front of `app`.

- [ ] **Step 1: Write the Caddyfile and compose service**

```
# Caddyfile
{$DOMAIN}

reverse_proxy app:8000
```

```yaml
# add to docker-compose.yml, under services:
  caddy:
    image: caddy:2.8
    depends_on:
      - app
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy-data:/data
    environment:
      - DOMAIN=${DOMAIN}
    restart: unless-stopped
```

**Note:** `reverse_proxy` sets `X-Forwarded-For` (what the API plan's rate limiter keys on). Host-loopback binding of `app:8000` doesn't affect compose-internal network traffic — Caddy reaches it via the `app` service hostname.

```yaml
# add caddy-data to the existing volumes: block in docker-compose.yml
  caddy-data:
```

Set `DOMAIN` in `.env` to your real domain (seeded by Task 2's `.env.example`) — a publicly-resolvable domain for Caddy's automatic HTTPS via Let's Encrypt; for purely local testing, skip this service and hit `app` directly on `localhost:8000`.

- [ ] **Step 2: Run and verify**

Run: `docker compose up -d caddy` (with `DOMAIN` pointed at a real domain whose DNS A record points at this host)
Expected: `curl -sI https://your-real-domain.example` returns `HTTP/2 200` with a valid Let's Encrypt certificate.

- [ ] **Step 3: Commit**

```bash
git add Caddyfile docker-compose.yml
git commit -m "feat: add Caddy TLS termination"
```

## Verification

1. `docker compose up -d opensearch app crawler` brings up the core stack; `docker compose ps` shows all three `running`.
2. `scripts/create_index.py` and `scripts/backfill_common_crawl.py` (Task 3) have been run at least once, and `curl -s http://localhost:9200/pages/_count` shows a non-zero count.
3. Open `http://localhost:8000/` (or the real domain through Caddy) in a browser and complete the Conversational UI plan's end-to-end verification (ask a question, see a streamed cited answer with sources and related-question chips).
4. Confirm the `local-llm` profile is genuinely optional: with it never started and `LLM_BASE_URL` pointed at a hosted API instead, the same UI flow still works.
