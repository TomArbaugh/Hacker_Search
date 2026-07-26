"""Layer 2 — OFFLINE: turn the JSONL corpus into a Chroma vector index.

Run this on your own machine after ingest.py:

    python build_index.py

It loads data/hn_items.jsonl, embeds each story with the sentence-transformer
model named in config, and writes a persisted Chroma collection to data/chroma/.
That directory IS the search index — the live app opens it read-only and never
rebuilds it. Re-running this script wipes and rebuilds the collection from
scratch, so it is safe to run whenever the corpus changes.

The first run downloads the embedding model (~90 MB for MiniLM); after that it
is cached locally by sentence-transformers.
"""
from __future__ import annotations

import json

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config

# Chroma's add() has an internal max batch size; stay well under it.
ADD_BATCH_SIZE = 1000
# Embed in batches so we don't hold every tensor in memory at once.
EMBED_BATCH_SIZE = 64


def load_records() -> list[dict]:
    """Read every story from the JSONL corpus."""
    if not config.ITEMS_PATH.exists():
        raise FileNotFoundError(
            f"{config.ITEMS_PATH} not found — run `python ingest.py` first."
        )
    records = []
    with open(config.ITEMS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def embed_text(rec: dict) -> str:
    """Build the string we actually embed for a story.

    Title carries most of the signal; story_text (Ask HN / text posts) adds
    context when present. This is the text the model 'reads' to place the story
    in vector space — keep it to what a human would skim.
    """
    title = rec.get("title", "")
    text = rec.get("text", "")
    return f"{title}\n\n{text}".strip() if text else title


def get_collection(client: chromadb.ClientAPI):
    """Return a fresh, empty collection, deleting any previous build.

    We use cosine distance because sentence-transformer embeddings are compared
    by angle, not magnitude — pair this with normalized embeddings below.
    """
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass  # nothing to delete on a first run
    return client.create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def build() -> int:
    """Embed the corpus and write the Chroma index. Returns records indexed."""
    records = load_records()
    print(f"Loaded {len(records)} stories from {config.ITEMS_PATH}")

    print(f"Loading embedding model: {config.EMBED_MODEL}")
    model = SentenceTransformer(config.EMBED_MODEL)

    documents = [embed_text(r) for r in records]

    # normalize_embeddings=True makes cosine distance behave correctly.
    print("Embedding stories...")
    embeddings = model.encode(
        documents,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = get_collection(client)

    # Metadata is what the app shows on the results page; the document string is
    # kept so we can build a text snippet without re-reading the JSONL.
    ids = [r["id"] for r in records]
    metadatas = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "author": r.get("author", ""),
            "points": r.get("points", 0),
            "num_comments": r.get("num_comments", 0),
        }
        for r in records
    ]

    print("Writing to Chroma...")
    for start in tqdm(range(0, len(ids), ADD_BATCH_SIZE), desc="Indexing", unit="batch"):
        end = start + ADD_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            embeddings=[e.tolist() for e in embeddings[start:end]],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return len(ids)


if __name__ == "__main__":
    count = build()
    print(f"Indexed {count} stories into {config.CHROMA_DIR}")
