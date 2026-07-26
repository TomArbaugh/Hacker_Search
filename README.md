---
title: Hacker Search
emoji: 🔍
colorFrom: orange
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Hacker_Search

Semantic search over Hacker News stories. Type a natural-language description of
an article ("a rust web framework that competes with actix") and get back the
stories that mean the same thing — not just the ones that share keywords.

## Goals & constraints

- **Free to build and run.** No paid APIs, no paid model hosting. Embeddings run
  locally on CPU; the vector store is on local disk.
- **Light at runtime.** All heavy work (pulling the corpus, embedding thousands
  of stories) happens *offline* on your own machine. The deployed app only embeds
  one short query string per request, so it fits a free host's RAM budget.

## Stack

| Layer        | Choice                                  | Why |
|--------------|-----------------------------------------|-----|
| HN data      | Algolia HN Search API + `requests`      | Free, no key, clean JSON — no scraping |
| Embeddings   | `sentence-transformers` (MiniLM-L6)     | Free, small, runs on CPU |
| Vector store | `chromadb`                              | Free, embedded, persists to disk |
| Web / auth   | `Flask` + `flask-login` + SQLite        | Free, simple, single-node |
| Reranker     | `bge-reranker-base` (optional, later)   | Better ordering when RAM allows |

## Architecture: offline index vs. live query

The expensive work happens **once, offline**. The server only does the cheap part.

```
┌─────────────── OFFLINE (your machine, run manually) ───────────────┐
│  ingest.py          build_index.py                                  │
│  Algolia HN API  →  load hn_items.jsonl                             │
│  paginate stories   → embed each (title + text) with MiniLM         │
│  write JSONL        → chroma.add(ids, embeddings, metadata)         │
│                     → persists to data/chroma/                      │
│         (commit / upload data/chroma to the host)                   │
└─────────────────────────────────────────────────────────────────────┘
                              │  deployed artifact, read-only
                              ▼
┌─────────────────── LIVE (free host, per request) ──────────────────┐
│  user query → search_core.search(q)                                 │
│      → embed ONE short string with MiniLM  (only ML work at runtime) │
│      → chroma.query(embedding, n=20) → nearest items + distances     │
│      → app.py renders results                                       │
└─────────────────────────────────────────────────────────────────────┘
```

Why this keeps it free: the server never ingests or bulk-embeds (the RAM/CPU-heavy
part lives offline), Chroma is opened read-only at runtime, and the embedding model
is loaded once at startup and kept warm.

## Module layout

```
Hacker_Search/
├── requirements.txt      # dependencies
├── .env                  # config overrides (gitignored)
├── .gitignore
│
├── config.py             # single source of truth for settings/paths
│
├── ingest.py             # OFFLINE: Algolia HN API → data/hn_items.jsonl
├── build_index.py        # OFFLINE: JSONL → embeddings → data/chroma/
│
├── search_core.py        # LIVE: embed query → query Chroma → return hits
├── app.py                # LIVE: public Flask search route + /healthz
│
├── Dockerfile            # HF Spaces image (bakes in model + index)
├── templates/            # Jinja templates (base, search)
│
└── data/                 # generated artifacts (big ones gitignored)
    ├── hn_items.jsonl    # raw pulled corpus (+ top_comments per story)
    └── chroma/           # persisted vector DB  ← the "index"
```

## Module responsibilities

- **`config.py`** — model name, data paths, top-K, Chroma collection name.
  Everything imports from here, so swapping MiniLM → bge is a one-line change.
- **`ingest.py`** — talks to Algolia only. Output: `hn_items.jsonl`, one story per
  line. Idempotent; safe to re-run to refresh the corpus.
- **`build_index.py`** — reads the JSONL, batches embeddings, writes Chroma.
  Re-runnable from scratch.
- **`search_core.py`** — `search(query, k) -> list[Hit]`. Loads model + opens
  Chroma at import time. No Flask imports (stays testable and reusable).
- **`app.py`** — a single public search route + `/healthz`, delegates to
  `search_core`. No accounts.

## Build order

1. **HN data** — `config.py` + `ingest.py` (get real data on disk).
2. **Index** — `build_index.py` (build + sanity-check the Chroma DB).
3. **Search** — `search_core.py` (nail search quality in isolation).
4. **Web** — `app.py` + templates (public, no login).

Each layer is verifiable before the next depends on it.

## Comment enrichment (planned)

Embedding the title alone (~10 words) is thin. We enrich each story's embed text
with a few of its **highest-ranked HN comments**, so the model understands what a
story is actually about — while guarding hard against low-quality/off-topic noise.

### Why this is safe from trolling
- HN already ranks comments by quality; trolls get downvoted and sink. We take
  only the **top few**, so we select from the pool trolls were filtered out of.
- We drop anything HN moderation killed (`dead` / `deleted` / flagged).
- The **title stays dominant** (see embed structure) — comments can only nudge a
  result's meaning, never hijack it. The real risk isn't trolls, it's an
  upvoted-but-off-topic joke; the title cap neutralizes it.

### Data source
Use the **Firebase HN API** (`/v0/item/{id}.json`), not Algolia, for comments.
A story item's `kids` array is HN's **ranked order** (best comment first), so we
never need per-comment scores (which HN doesn't expose anyway) — we just take the
front of `kids`.

### Filter rules (applied in order, per story)
1. Fetch the story item; read its `kids` (top-level comments, rank-ordered).
2. Walk `kids` in order, fetching each comment item, until we have
   `COMMENTS_PER_STORY` keepers. For each candidate, **skip** if:
   - `type != "comment"`, or `deleted` / `dead` is true (moderation filter)
   - the text (HTML stripped) has fewer than `COMMENT_MIN_WORDS` words
     (drops "This.", "+1", low-signal one-liners)
3. **Top-level only** — do not recurse into replies (lower-ranked, tangential).
4. Truncate each kept comment to `COMMENT_MAX_CHARS` at a word boundary.

### Embed-text structure (title-dominant)
```
<title>

<comment 1>
<comment 2>
<comment 3>
```
The title is always first and never truncated. Comments follow, capped so their
combined length can't overwhelm the title. Comments are used **for embedding
only** — never displayed in the UI (avoids copyright + keeps the results clean).
The snippet shown to users still comes from the story's own text.

### Config knobs (added to config.py)
- `COMMENTS_ENABLED` (default true) — master switch, so title-only is one flag away
- `COMMENTS_PER_STORY` (default 3) — how many top comments to keep
- `COMMENT_MIN_WORDS` (default 15) — junk/one-liner filter
- `COMMENT_MAX_CHARS` (default 300) — per-comment truncation
- `HN_FIREBASE_BASE` — Firebase API base URL

### Cost & caching
This adds one request per story plus one per kept comment — the slowest part of
ingest. Comments are stored in `hn_items.jsonl` (a `top_comments` field on each
record), so `build_index.py` never re-fetches, and re-running the index is fast.

### Validate before committing
Rebuilding is cheap, so A/B it on the same 5–10 queries: (1) title only,
(2) title + comments. Keep comments only if they measurably help *your* corpus.

## Running (offline steps, once)

```bash
pip install -r requirements.txt
python ingest.py          # pull HN stories → data/hn_items.jsonl
python build_index.py     # embed + index → data/chroma/
```

## Running the app

```bash
flask --app app run
```
