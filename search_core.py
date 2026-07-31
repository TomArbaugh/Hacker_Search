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

import logging
import math
import re
import sys
import time
from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)
# Ensure output goes to stdout
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

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
    logger.info(f"[_DENSE] Starting, embedding query")
    print(f"[_DENSE] Starting, embedding query", flush=True)
    
    embedding = _model.encode([query], normalize_embeddings=True)[0].tolist()
    
    logger.info(f"[_DENSE] Embedding complete, calling _collection.query() with n={n}")
    print(f"[_DENSE] Embedding complete, calling _collection.query() with n={n}", flush=True)
    
    res = _collection.query(
        query_embeddings=[embedding], n_results=n, include=["distances"]
    )
    
    logger.info(f"[_DENSE] _collection.query() returned, processing results")
    print(f"[_DENSE] _collection.query() returned, processing results", flush=True)
    
    ids = res["ids"][0]
    dists = res["distances"][0]
    
    logger.info(f"[_DENSE] Complete, returning {len(ids)} results")
    print(f"[_DENSE] Complete, returning {len(ids)} results", flush=True)
    
    return ids, dict(zip(ids, dists))


def _sparse_ids(query: str, n: int) -> list[str]:
    """Keyword retrieval: return up to n story ids ranked by BM25.

    Documents with zero term overlap are dropped so they don't add rank noise to
    the fusion.
    """
    logger.info(f"[_SPARSE] Starting BM25 search")
    print(f"[_SPARSE] Starting BM25 search", flush=True)
    
    if _bm25 is None:
        logger.info(f"[_SPARSE] BM25 is None, returning empty list")
        return []
    
    logger.info(f"[_SPARSE] Tokenizing query")
    print(f"[_SPARSE] Tokenizing query", flush=True)
    tokens = _tokenize(query)
    logger.info(f"[_SPARSE] Query tokens: {tokens}")
    
    logger.info(f"[_SPARSE] Calling _bm25.get_scores()")
    print(f"[_SPARSE] Calling _bm25.get_scores()", flush=True)
    scores = _bm25.get_scores(tokens)
    logger.info(f"[_SPARSE] Got scores array of length {len(scores)}")
    print(f"[_SPARSE] Got scores array of length {len(scores)}", flush=True)
    
    logger.info(f"[_SPARSE] Sorting scores")
    print(f"[_SPARSE] Sorting scores", flush=True)
    # Indices of the top-n scores, best first, keeping only positive overlap.
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    logger.info(f"[_SPARSE] Building result list")
    print(f"[_SPARSE] Building result list", flush=True)
    result = [_ids[i] for i in ranked[:n] if scores[i] > 0]
    
    logger.info(f"[_SPARSE] Complete, returning {len(result)} results")
    print(f"[_SPARSE] Complete, returning {len(result)} results", flush=True)
    
    return result


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
    logger.info(f"[SEARCH] Starting search for query: '{query}'")
    print(f"[SEARCH] Starting search for query: '{query}'", flush=True)
    
    query = (query or "").strip()
    if not query:
        logger.info("[SEARCH] Empty query, returning []")
        return []

    logger.info(f"[SEARCH] Calling _dense() with n={config.HYBRID_CANDIDATES}")
    print(f"[SEARCH] Calling _dense() with n={config.HYBRID_CANDIDATES}", flush=True)
    dense, dense_dist = _dense(query, config.HYBRID_CANDIDATES)
    logger.info(f"[SEARCH] _dense() returned {len(dense)} results")
    print(f"[SEARCH] _dense() returned {len(dense)} results", flush=True)
    
    if config.HYBRID_ENABLED:
        logger.info(f"[SEARCH] Calling _sparse_ids() with n={config.HYBRID_CANDIDATES}")
        print(f"[SEARCH] Calling _sparse_ids() with n={config.HYBRID_CANDIDATES}", flush=True)
        sparse = _sparse_ids(query, config.HYBRID_CANDIDATES)
        logger.info(f"[SEARCH] _sparse_ids() returned {len(sparse)} results")
        print(f"[SEARCH] _sparse_ids() returned {len(sparse)} results", flush=True)
    else:
        sparse = []
        logger.info("[SEARCH] Hybrid disabled, skipping sparse search")

    logger.info(f"[SEARCH] Calling _rrf() with dense={len(dense)}, sparse={len(sparse)}")
    print(f"[SEARCH] Calling _rrf() with dense={len(dense)}, sparse={len(sparse)}", flush=True)
    fused = _rrf([dense, sparse], config.RRF_K)
    logger.info(f"[SEARCH] _rrf() returned {len(fused)} fused results")
    print(f"[SEARCH] _rrf() returned {len(fused)} fused results", flush=True)
    
    if not fused:
        logger.info("[SEARCH] No fused results, returning []")
        return []

    logger.info("[SEARCH] Starting scoring loop")
    print("[SEARCH] Starting scoring loop", flush=True)
    now = time.time()
    scored = []
    for doc_id, rrf_score in fused.items():
        meta = _meta_by_id.get(doc_id, {})
        final = rrf_score * _boost(meta, now)
        scored.append((final, doc_id, meta))

    logger.info(f"[SEARCH] Scored {len(scored)} results, now sorting")
    print(f"[SEARCH] Scored {len(scored)} results, now sorting", flush=True)
    scored.sort(key=lambda t: t[0], reverse=True)

    logger.info(f"[SEARCH] Building Hit objects for top {k} results")
    print(f"[SEARCH] Building Hit objects for top {k} results", flush=True)
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
    
    logger.info(f"[SEARCH] Completed! Returning {len(hits)} hits")
    print(f"[SEARCH] Completed! Returning {len(hits)} hits", flush=True)
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
