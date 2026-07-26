"""Layer 3 — LIVE: the reusable search function the app calls per request.

This is the heart of the app: `search(query)` embeds one short query string,
asks Chroma for the nearest stories, and returns clean result objects. It has
NO Flask imports on purpose — that keeps it independently testable and reusable
(you can call search() from a script, a notebook, or a different web framework).

Cost model: the embedding model and the Chroma collection are loaded ONCE at
import time and kept warm for the process lifetime. A request then only embeds a
single short string, so the per-query work is tiny — which is what lets the app
run on a free host. Chroma is opened read-only; this module never writes.
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

import config

# Longest snippet (characters) we show under a result title.
SNIPPET_MAX = 200


@dataclass
class Hit:
    """One search result, shaped for direct rendering in a template."""
    number: int          # 1-based rank in the result list
    title: str
    url: str
    author: str
    points: int
    num_comments: int
    snippet: str         # short body excerpt (empty for pure link posts)
    distance: float      # cosine distance; lower = closer in meaning


# --- Loaded once at import, kept warm ----------------------------------------
_model = SentenceTransformer(config.EMBED_MODEL)


def _get_collection():
    """Open the persisted Chroma collection read-only.

    Raises a clear error if the index hasn't been built yet, since that is the
    most common setup mistake.
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        return client.get_collection(config.CHROMA_COLLECTION)
    except Exception as exc:  # collection missing / never built
        raise RuntimeError(
            f"Chroma collection '{config.CHROMA_COLLECTION}' not found at "
            f"{config.CHROMA_DIR}. Run `python build_index.py` first."
        ) from exc


_collection = _get_collection()


def _snippet(document: str, title: str) -> str:
    """Build a short excerpt from a story's embedded document.

    The document we indexed is "title\\n\\ntext", so we strip the leading title
    to avoid repeating it, then truncate. Pure link posts have no body and yield
    an empty snippet.
    """
    body = document
    if title and body.startswith(title):
        body = body[len(title):]
    body = body.strip()
    if not body:
        return ""
    if len(body) <= SNIPPET_MAX:
        return body
    return body[:SNIPPET_MAX].rsplit(" ", 1)[0] + "…"  # trim at a word, add …


def search(query: str, k: int = config.TOP_K) -> list[Hit]:
    """Return up to `k` stories most similar in meaning to `query`.

    Returns an empty list for an empty/blank query rather than raising, so
    callers (like the web form) can pass user input straight through.
    """
    query = (query or "").strip()
    if not query:
        return []

    query_embedding = _model.encode([query], normalize_embeddings=True)[0].tolist()
    res = _collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["metadatas", "documents", "distances"],
    )

    # Chroma returns each field as a list-of-lists (one inner list per query).
    metadatas = res["metadatas"][0]
    documents = res["documents"][0]
    distances = res["distances"][0]

    hits: list[Hit] = []
    for i, (meta, doc, dist) in enumerate(zip(metadatas, documents, distances), start=1):
        title = meta.get("title", "")
        hits.append(
            Hit(
                number=i,
                title=title,
                url=meta.get("url", ""),
                author=meta.get("author", ""),
                points=meta.get("points", 0),
                num_comments=meta.get("num_comments", 0),
                snippet=_snippet(doc or "", title),
                distance=float(dist),
            )
        )
    return hits


if __name__ == "__main__":
    # Tiny CLI so you can sanity-check search quality without the web app:
    #     python search_core.py "a rust web framework"
    import sys

    q = " ".join(sys.argv[1:]) or "privacy risks in consumer gadgets"
    print(f'Query: "{q}"\n')
    for hit in search(q, k=5):
        print(f"[{hit.number}] {hit.distance:.3f}  {hit.title}")
        if hit.snippet:
            print(f"      {hit.snippet}")
        if hit.url:
            print(f"      {hit.url}")
