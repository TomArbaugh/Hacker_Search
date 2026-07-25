"""Layer 1 — OFFLINE: pull Hacker News stories into a local JSONL corpus.

Run this on your own machine (not the server):

    python ingest.py

It talks to the free Algolia HN Search API and writes one JSON object per line
to data/hn_items.jsonl. It is idempotent: re-running overwrites the file with a
fresh pull. Nothing here embeds or indexes — that is Layer 2 (build_index.py).

How pagination works: Algolia caps a single query at 1000 results, so we cannot
just ask for page 2, 3, 4... forever. Instead we page *backwards through time*:
each request asks for stories older than the oldest one we have so far, using
the created_at_i (unix seconds) cursor. That lets us walk as far back as we like.
"""
from __future__ import annotations  # allow `int | None` annotations on Python 3.9

import json
import time

import requests
from tqdm import tqdm

import config


def _fetch_page(before_ts: int | None) -> list[dict]:
    """Fetch one page of stories older than `before_ts` (None = newest first)."""
    params = {
        "tags": "story",
        "hitsPerPage": config.INGEST_PAGE_SIZE,
        # Only stories with enough points to be worth indexing.
        "numericFilters": f"points>={config.INGEST_MIN_POINTS}",
    }
    if before_ts is not None:
        # Add the time cursor: strictly older than the oldest we've seen.
        params["numericFilters"] += f",created_at_i<{before_ts}"

    resp = requests.get(
        f"{config.HN_API_BASE}/search_by_date",  # sorted newest-first by date
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("hits", [])


def _clean(hit: dict) -> dict | None:
    """Normalize an Algolia hit into the record shape we store.

    Returns None for hits with no usable text to embed.
    """
    title = (hit.get("title") or "").strip()
    if not title:
        return None

    return {
        "id": str(hit["objectID"]),
        "title": title,
        # story_text is populated for Ask HN / text posts; often empty for links.
        "text": (hit.get("story_text") or "").strip(),
        "url": hit.get("url") or "",
        "author": hit.get("author") or "",
        "points": hit.get("points") or 0,
        "num_comments": hit.get("num_comments") or 0,
        "created_at_i": hit.get("created_at_i") or 0,
    }


def ingest() -> int:
    """Pull up to config.INGEST_TARGET stories and write them to ITEMS_PATH.

    Returns the number of records written.
    """
    seen_ids: set[str] = set()
    records: list[dict] = []
    cursor: int | None = None  # created_at_i to page backwards from

    with tqdm(total=config.INGEST_TARGET, desc="Fetching HN stories", unit="story") as bar:
        while len(records) < config.INGEST_TARGET:
            hits = _fetch_page(cursor)
            if not hits:
                break  # ran out of stories

            oldest_ts = cursor
            new_this_page = 0
            for hit in hits:
                ts = hit.get("created_at_i")
                if ts is not None:
                    # Track the oldest timestamp to advance the cursor.
                    oldest_ts = ts if oldest_ts is None else min(oldest_ts, ts)

                rec = _clean(hit)
                if rec is None or rec["id"] in seen_ids:
                    continue
                seen_ids.add(rec["id"])
                records.append(rec)
                new_this_page += 1
                bar.update(1)
                if len(records) >= config.INGEST_TARGET:
                    break

            # Guard against getting stuck: if a page added nothing new or the
            # cursor didn't move backwards, stop rather than loop forever.
            if new_this_page == 0 or oldest_ts == cursor:
                break
            cursor = oldest_ts

            time.sleep(config.INGEST_REQUEST_DELAY)  # be polite to the free API

    with open(config.ITEMS_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records)


if __name__ == "__main__":
    count = ingest()
    print(f"Wrote {count} stories to {config.ITEMS_PATH}")
