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
├── users.py              # LIVE: SQLite auth (init_db, get_user_by_id, ...)
├── app.py                # LIVE: Flask routes + UI
│
├── templates/            # Jinja templates (base, search, login, signup)
│
└── data/                 # generated artifacts (big ones gitignored)
    ├── hn_items.jsonl    # raw pulled corpus
    ├── chroma/           # persisted vector DB  ← the "index"
    └── users.db          # SQLite user table
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
- **`users.py`** — SQLite auth; passwords hashed with `werkzeug.security`.
- **`app.py`** — routes only, delegates to `search_core` and `users`.

## Build order

1. **HN data** — `config.py` + `ingest.py` (get real data on disk).
2. **Index** — `build_index.py` (build + sanity-check the Chroma DB).
3. **Search** — `search_core.py` (nail search quality in isolation).
4. **Web** — `users.py` + `app.py` + templates (wire up the UI last).

Each layer is verifiable before the next depends on it.

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
