# Full-Stack Conversational Search Engine — Design Document

## Context

Goal: an ad-free, non-commercially-biased search engine that answers queries directly with fully-sourced, cited answers (not just a link list) — good for both human users and AI agents — while reaching legal public web content that mainstream engines under-index ("deep web" = niche/long-tail sites, not the dark web).

The design evolved through this session:
1. Started as a plain ad-free, deep-web-reaching search engine (crawler + indexer + API).
2. Expanded to a conversational, Perplexity-style engine: synthesized cited answers, related-question suggestions, and a chat interface — after researching Perplexity, You.com, Phind, Kagi Assistant, agent-facing search APIs (Exa, Tavily, Brave, SearXNG), and RAG grounding/citation techniques.
3. One boundary was set during design: the answer-synthesis layer will use standard, safety-tuned LLMs (hosted or self-hosted) — not "abliterated"/safety-stripped models. That request was declined because it targets a different axis (removing refusals on genuinely harmful content) than what every other requirement in this doc is about (search neutrality — no ad bias, no editorial reranking, broad legal coverage). Combined with a public, non-bot-filtered API, safety-stripped generation would remove the only gate between "anyone/any agent asks for something harmful" and the system producing it. Everything else in this document proceeded as requested.

## Key Decisions

- **Retrieval**: hybrid of a purpose-built crawler (targeted at niche/long-tail domains) and the free/open Common Crawl corpus, feeding a self-hosted index. Chosen over a from-scratch general crawler (unrealistic at Google/Bing scale for a self-hosted project) and over a pure meta-search aggregator (wouldn't add real deep-web coverage of its own — it only re-serves what other engines already index).
- **Synthesis backend**: model-agnostic `LLMClient` behind an OpenAI-compatible chat-completions interface — works with any backend (Claude, GPT, Gemini, or a self-hosted open-weight instruct model via vLLM/Ollama) via config, not hardcoded to one vendor. No safety-stripped models (see Context above).
- **Reference architecture**: Perplexica (github.com/ItzCrazyKns/Perplexica, MIT-licensed, SearXNG + LangChain + cited LLM answers) is a proven open-source implementation of most of this pattern and is a strong basis to adapt from rather than building the RAG pipeline from zero.
- **Stack**: Python 3.12 — `httpx`/`asyncio` + Scrapling (crawler), `boto3`/`warcio` (Common Crawl ingestion), self-hosted OpenSearch (Apache 2.0, index + vector store), FastAPI (API), server-rendered HTML/htmx + minimal JS for streaming (frontend).
- **Hard constraints (unchanged throughout)**: no dark web/onion sites; no ads or paid placement; no proprietary paid APIs for crawling/indexing specifically; respects `robots.txt` and crawl-delay etiquette; the only content exclusions are a legal floor (CSAM hash-matching, DMCA takedown compliance) — no editorial/political/brand-safety curation beyond that.

## 1. Architecture Overview

```
                     ┌─────────────────────┐
                     │   Common Crawl (S3)  │  free, open, petabyte-scale
                     │  monthly WARC dumps  │  public web corpus
                     └──────────┬───────────┘
                                │ (fetch + parse relevant WARC/WET records)
                                ▼
 ┌──────────────┐      ┌───────────────────┐
 │  Own Crawler  │─────▶│   Ingest/Parse     │  extract text, strip boilerplate,
 │ (niche/deep-  │      │   Pipeline         │  dedupe, detect ads/sponsored blocks
 │  web targets) │      └─────────┬─────────┘
 └──────────────┘                │
                                  ▼
                        ┌───────────────────┐
                        │   Index Store      │  OpenSearch — self-hosted,
                        │ (inverted index +  │  no proprietary API
                        │  embeddings)        │
                        └─────────┬─────────┘
                                  │
                                  ▼
                        ┌───────────────────┐
                        │ Retrieve + Rerank   │  hybrid BM25 + embedding search,
                        │                      │  cross-encoder rerank top-k
                        └─────────┬─────────┘
                                  │
                                  ▼
                        ┌───────────────────┐
                        │ Answer Synthesis     │  LLMClient (any backend) generates
                        │ + Verification        │  cited answer; post-gen check
                        │                      │  drops unsupported sentences
                        └─────────┬─────────┘
                                  │
                     ┌────────────┴────────────┐
                     ▼                          ▼
           ┌──────────────────┐      ┌──────────────────┐
           │   Public API       │      │ Conversational UI  │
           │ (human + AI-agent  │      │ (chat, streaming,  │
           │  friendly, no      │      │  citations, related│
           │  CAPTCHA/rate-wall)│      │  questions)        │
           └──────────────────┘      └──────────────────┘
```

## 2. Crawler Design

Four components, scoped for a niche/deep-web crawler rather than a general web-scale one:

- **Seed Manager** — curated list of target domains/topics, plus a discovery queue for links found while crawling.
- **Frontier/Scheduler** — per-domain queue, `robots.txt` compliance and crawl-delay enforcement (hard requirement).
- **Fetcher** — async HTTP (`httpx`; Scrapling for JS-rendered pages).
- **Extractor** — clean text extraction; also where ad-filtering happens (strip ad-network markup, sponsored blocks, boilerplate before indexing).

```python
AD_PATTERNS = re.compile(r"(doubleclick|googlesyndication|sponsor(ed)?|advert)", re.I)

def extract_clean_text(html: str, url: str) -> ExtractedPage:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "iframe", "ins"]):
        if tag.get("class") and AD_PATTERNS.search(" ".join(tag["class"])):
            tag.decompose()
        elif tag.name == "iframe" and AD_PATTERNS.search(tag.get("src", "")):
            tag.decompose()
    body = trafilatura.extract(str(soup), url=url)  # boilerplate removal
    return ExtractedPage(url=url, text=body, ad_ratio=estimate_ad_density(soup))
```

A per-page `ad_ratio`/thin-content score is stored alongside extracted text and feeds ranking later, so ad-farm/content-mill domains are deprioritized systemically.

**Common Crawl ingestion**: no fetching needed — pull the monthly WARC/WET index from the public S3 bucket, filter to target domains/topics, run the same `extract_clean_text` step on WET (pre-extracted text) records.

## 3. Indexer

Self-hosted OpenSearch. Per-document fields:

| Field | Purpose |
|---|---|
| `url`, `domain`, `crawl_date` | identity/freshness |
| `title`, `body_text` | BM25 full-text match |
| `chunks[]` (~450 tokens, 50-token overlap) | retrieval unit for the synthesis stage, each chunk keeps url/title metadata |
| `embedding` (vector, per chunk) | semantic/dense retrieval |
| `ad_ratio`, `domain_quality` | ranking signal only, never shown to users |
| `simhash` | near-duplicate detection at index time |

```python
def index_page(client: OpenSearch, page: ExtractedPage):
    if is_near_duplicate(client, page.simhash):
        return
    for chunk in chunk_text(page.text, size=450, overlap=50):
        client.index(index="pages", body={
            "url": page.url, "domain": page.domain, "title": page.title,
            "chunk_text": chunk, "embedding": embed(chunk),
            "ad_ratio": page.ad_ratio, "crawl_date": page.crawl_date,
        })
```

## 4. Retrieval, Ranking & "Uncensored" as Engineering Rules

**Ranking inputs are an explicit allow-list**: lexical match (BM25), semantic similarity (embedding cosine), domain-quality/spam signal, near-duplicate penalty, freshness. Never: advertiser bids, partner deals, manually-assigned "trust scores", or user identity/browsing history.

```python
def score(query_emb, chunk) -> float:
    return (
        0.35 * bm25(query, chunk.chunk_text)
        + 0.30 * cosine(query_emb, chunk.embedding)
        + 0.20 * chunk.domain_quality
        - 0.15 * chunk.ad_ratio
        + freshness_decay(chunk.crawl_date)
    )
```

- Retrieve ~20-30 candidate chunks this way, then **rerank with a cross-encoder** down to the top 5-8 — the set that gets passed to answer synthesis.
- **Domain diversity cap** — max ~2 chunks per domain in the reranked set, so large SEO-optimized domains can't crowd out independent/niche sources.
- **No manual re-ranking**, with one narrow legal exception: a hard exclusion list checked at index time for CSAM (hash-matching against NCMEC/PhotoDNA-style hash sets) and DMCA compliance. Nothing else is manually removed or reordered.

## 5. Answer Synthesis & Verification

The core RAG pattern used by Perplexity, You.com, Kagi, and the agent-facing search APIs researched: retrieve → rerank → **generate only from the provided sources, with forced inline citations** → verify.

```python
SYSTEM_PROMPT = """Answer using ONLY the numbered sources below.
Cite every claim inline as [n]. If the sources don't cover the question,
say "not enough information" rather than guessing."""

async def synthesize_answer(query: str, chunks: list[Chunk], llm: LLMClient) -> Answer:
    sources_block = "\n".join(f"[{i+1}] {c.chunk_text} (source: {c.url})"
                               for i, c in enumerate(chunks))
    raw = await llm.chat(
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"{sources_block}\n\nQuestion: {query}"}],
        temperature=0.1, stream=True,
    )
    return verify_citations(raw, chunks)  # drops/flags sentences that don't
                                           # match their cited chunk's content
```

**`LLMClient` is a model-agnostic interface** speaking the OpenAI-compatible chat-completions schema — the same schema Perplexity's Sonar API, most hosted providers, and self-hosted servers (vLLM, Ollama, LM Studio) already implement. Swapping backends (Claude, GPT, Gemini, or a self-hosted open-weight instruct model) is a config change (model name + base URL + API key), not a code change. Only standard, safety-tuned instruct models are in scope — see Context for why safety-stripped ("abliterated") models were explicitly excluded.

**`verify_citations`** is the accuracy safeguard: after generation, each sentence is embedding-matched back against the chunk it cites; sentences below a similarity threshold are flagged as unsupported and dropped before the user sees them. This is what makes "accuracy prioritized, fully sourced" enforced rather than aspirational.

**Related questions**: one small follow-up LLM call after the answer, or a template pulling from co-occurring queries.

## 6. API — Human & AI-Agent Friendly

Two endpoints, both usable by browsers or agents identically — no CAPTCHA, no JS challenge, no identity-based gating. A simple per-IP/per-key rate limit protects infrastructure; that's fair-use throttling, not bot detection.

```python
@app.get("/api/v1/search")
async def search(q: str, limit: int = 10, include_fulltext: bool = False):
    hits = retrieve_and_rerank(q, limit=limit)
    return {"query": q, "results": [
        {"url": h.url, "title": h.title, "snippet": h.snippet,
         "domain": h.domain, "published": h.crawl_date,
         **({"fulltext": h.chunk_text} if include_fulltext else {})}
        for h in hits
    ]}

@app.post("/api/v1/answer")
async def answer(query: str, conversation_id: str | None = None):
    chunks = retrieve_and_rerank(query, limit=8, context=conversation_id)
    result = await synthesize_answer(query, chunks, llm=default_llm_client)
    return StreamingResponse(result.token_stream())  # sources + related_questions
                                                       # sent as a final SSE event
```

FastAPI serves an OpenAPI schema at `/openapi.json` automatically for agent self-discovery.

## 7. Conversational UI

```
┌────────────────────────────────────────────────┐
│  🔍  ask anything...                    [Send]  │
├────────────────────────────────────────────────┤
│  You: What causes the northern lights?           │
│                                                    │
│  Answer: The aurora borealis occurs when charged  │
│  particles from the sun collide with gases in     │
│  Earth's atmosphere [1][2]. Oxygen produces green │
│  and red light, nitrogen produces blue/purple [3].│
│                                                    │
│  Sources: [1] spaceweather.org  [2] noaa.gov      │
│           [3] independent astronomy blog           │
│               (deep-web source, low domain rank)   │
│                                                    │
│  Related: • Why do auroras have different colors?  │
│           • Can auroras be seen from the equator?  │
│                                                    │
│  [Type a follow-up...]                            │
└────────────────────────────────────────────────┘
```

Server-rendered HTML (Jinja2) + htmx for pagination/interactions, plus minimal vanilla JS for SSE token streaming (the one place the "no heavy JS" default is relaxed, since streaming is core to the experience). `conversation_id` ties turns together server-side so follow-ups retain context. Source cards are hoverable/expandable and transparently note when a source is a niche/deep-web find vs. a mainstream domain. Related-question chips start the next turn.

## 8. Deployment

Docker Compose, four services: OpenSearch (index), the FastAPI app (serves API + HTML frontend), a scheduled crawler job (cron-triggered container), and the LLM backend (either external API — no extra service — or a local vLLM/Ollama container if self-hosting synthesis). Bring up OpenSearch → run the one-time Common Crawl backfill → start the crawler on a schedule → bring up the app behind Caddy/nginx for TLS. Self-hostable on a single mid-size VM to start (plus a GPU box if self-hosting the LLM).

## Future Enhancements (discussed, deliberately out of this design's core scope)

- **Live meta-search aggregation (SearXNG-style)** — fanning queries out to other search engines at query time for very recent/breaking content the own-crawler/Common-Crawl index hasn't caught up to yet. Raised during design as a phase-2 option; deferred from the core design because research showed it's fragile (inherits upstream engines' ToS/anti-bot/HTML changes) and doesn't add genuine deep-web coverage on its own — it only re-serves what those upstream engines already index. The retrieval layer in Section 4 is structured so this could be added later as an extra candidate source alongside the own index, without changing the rerank/synthesis stages downstream.

## Constraints Verification

- ✅ No ads/paid placement — ranking allow-list has no advertiser-bid input.
- ✅ No proprietary paid APIs for crawling/indexing — Common Crawl, OpenSearch, self-built crawler are all open. (The synthesis LLM may optionally be a paid API by user choice — a separate layer, not crawling/indexing.)
- ✅ Respects `robots.txt` and crawl-delay etiquette — enforced in the Frontier/Scheduler, non-optional.
- ✅ No illegal/private content, no dark web/onion sites — CSAM hash-matching and DMCA compliance are the only content exclusions; no auth-bypass or Tor functionality anywhere in the design.
- ✅ No safety-stripped ("abliterated") models — synthesis backend is model-agnostic but restricted to standard, safety-tuned instruct models.
- ✅ Design + code only in this document; implementation is planned separately.
