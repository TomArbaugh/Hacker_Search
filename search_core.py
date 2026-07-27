"""Layer 3 — LIVE: the reusable search function the app calls per request.

This is the heart of the app: `search(query)` runs hybrid retrieval — dense
(vector) + sparse (BM25 keyword) — fuses the two with Reciprocal Rank Fusion,
applies a light popularity/recency boost, and returns clean result objects. It
has NO Flask imports on purpose, so it stays independently testable and reusable.

Cost model: the embedding model, the Chroma collection, and the in-memory BM25
index are all built ONCE at import time and kept warm. A request then only embeds
one short query string and scores it — tiny per-query work, which is what lets the
app run on a free host. Chroma is opened read-only; this module never writes.

Set HYBRID_ENABLED=false (and the boosts to 0) for pure semantic search.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config

# Longest snippet (characters) we show under a result title.
SNIPPET_MAX = 200

# Simple word tokenizer for BM25: lowercase alphanumeric runs.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


@dataclass
class Hit:
    """One search result, shaped for direct rendering in a template."""
    number: int          # 1-based rank in the result list
    title: str
    url: str
    author: str
    points: int
    num_comments: int
    snippet: str            # short body excerpt (empty for pure link posts)
    distance: float | None  # dense cosine distance (lower=closer); None if keyword-only


# --- Loaded once at import, kept warm ----------------------------------------
_model = SentenceTransformer(config.EMBED_MODEL)


def _get_collection():
    """Open the persisted Chroma collection read-only, with a clear error if the
    index hasn't been built yet (the most common setup mistake)."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        return client.get_collection(config.CHROMA_COLLECTION)
    except Exception as exc:
        raise RuntimeError(
            f"Chroma collection '{config.CHROMA_COLLECTION}' not found at "
            f"{config.CHROMA_DIR}. Run `python build_index.py` first."
        ) from exc


_collection = _get_collection()

# Load the whole corpus (ids + metadata) into memory once so we can build the
# BM25 index and look up records by id during fusion/boosting. Fine at this
# scale (a few thousand docs); revisit if the corpus grows into the millions.
_corpus = _collection.get(include=["metadatas"])
_ids: list[str] = _corpus["ids"]
_meta_by_id: dict[str, dict] = dict(zip(_ids, _corpus["metadatas"]))

# BM25 index over each story's OWN words (title + body) — not comments, so
# keyword matching stays precise. Aligned positionally with _ids.
_bm25_docs = [
    _tokenize(f"{m.get('title', '')} {m.get('text', '')}")
    for m in _corpus["metadatas"]
]
_bm25 = BM25Okapi(_bm25_docs) if _bm25_docs else None

# Precompute the corpus max of log1p(points) so the popularity boost is in [0,1].
_max_log_points = max(
    (math.log1p(m.get("points", 0) or 0) for m in _corpus["metadatas"]),
    default=0.0,
) or 1.0


def _snippet(text: str) -> str:
    """Build a short excerpt from a story's OWN body text.

    Sourced from the story's `text` metadata, never from comments (which are
    embedded for search but must not be displayed). Link posts have no body and
    yield an empty snippet (the UI hides the line).
    """
    body = (text or "").strip()
    if not body:
        return ""
    if len(body) <= SNIPPET_MAX:
        return body
    return body[:SNIPPET_MAX].rsplit(" ", 1)[0] + "…"


def _dense(query: str, n: int) -> tuple[list[str], dict[str, float]]:
    """Vector retrieval: return (ranked ids, {id: cosine distance})."""
    embedding = _model.encode([query], normalize_embeddings=True)[0].tolist()
    res = _collection.query(
        query_embeddings=[embedding], n_results=n, include=["distances"]
    )
    ids = res["ids"][0]
    dists = res["distances"][0]
    return ids, dict(zip(ids, dists))


def _sparse_ids(query: str, n: int) -> list[str]:
    """Keyword retrieval: return up to n story ids ranked by BM25.

    Documents with zero term overlap are dropped so they don't add rank noise to
    the fusion.
    """
    if _bm25 is None:
        return []
    scores = _bm25.get_scores(_tokenize(query))
    # Indices of the top-n scores, best first, keeping only positive overlap.
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [_ids[i] for i in ranked[:n] if scores[i] > 0]


def _rrf(ranked_lists: list[list[str]], k: int) -> dict[str, float]:
    """Reciprocal Rank Fusion: combine ranked id lists by position, not score."""
    fused: dict[str, float] = {}
    for ids in ranked_lists:
        for rank, doc_id in enumerate(ids):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def _boost(meta: dict, now: float) -> float:
    """Light multiplicative tie-breaker from popularity + recency (>= 1.0)."""
    points = meta.get("points", 0) or 0
    popularity = math.log1p(points) / _max_log_points  # in [0, 1]

    created = meta.get("created_at_i", 0) or 0
    recency = 0.0
    if created and config.RECENCY_HALFLIFE_DAYS > 0:
        age_days = max(0.0, (now - created) / 86400.0)
        recency = 0.5 ** (age_days / config.RECENCY_HALFLIFE_DAYS)  # in (0, 1]

    return 1.0 + config.POINTS_BOOST * popularity + config.RECENCY_BOOST * recency


def search(query: str, k: int = config.TOP_K) -> list[Hit]:
    """Return up to `k` stories most relevant to `query`.

    Hybrid: dense + BM25, fused with RRF, then a small popularity/recency boost.
    With HYBRID_ENABLED=false it degrades to dense-only. Returns [] for a blank
    query so callers can pass user input straight through.
    """
    query = (query or "").strip()
    if not query:
        return []

    dense, dense_dist = _dense(query, config.HYBRID_CANDIDATES)
    sparse = _sparse_ids(query, config.HYBRID_CANDIDATES) if config.HYBRID_ENABLED else []

    fused = _rrf([dense, sparse], config.RRF_K)
    if not fused:
        return []

    now = time.time()
    scored = []
    for doc_id, rrf_score in fused.items():
        meta = _meta_by_id.get(doc_id, {})
        final = rrf_score * _boost(meta, now)
        scored.append((final, doc_id, meta))

    scored.sort(key=lambda t: t[0], reverse=True)

    hits: list[Hit] = []
    for i, (_, doc_id, meta) in enumerate(scored[:k], start=1):
        hits.append(
            Hit(
                number=i,
                title=meta.get("title", ""),
                url=meta.get("url", ""),
                author=meta.get("author", ""),
                points=meta.get("points", 0),
                num_comments=meta.get("num_comments", 0),
                snippet=_snippet(meta.get("text", "")),
                # Real cosine distance when the dense retriever found it; None for
                # keyword-only hits (matched by BM25 but outside the dense top-N).
                distance=dense_dist.get(doc_id),
            )
        )
    return hits


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "privacy risks in consumer gadgets"
    print(f'Query: "{q}"  (hybrid={config.HYBRID_ENABLED})\n')
    for hit in search(q, k=5):
        tag = "kw-only" if hit.distance is None else f"d={hit.distance:.3f}"
        print(f"[{hit.number}] ({tag})  {hit.title}")
        if hit.snippet:
            print(f"      {hit.snippet}")
