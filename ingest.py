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

import html
import json
import re
import time

import requests
from tqdm import tqdm

import config

# Strip HTML tags from comment bodies (Firebase returns comment text as HTML).
_TAG_RE = re.compile(r"<[^>]+>")


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


def _strip_html(text: str) -> str:
    """Turn HN's HTML comment text into plain text.

    HN comments arrive as HTML (<p>, <a>, entities like &#x27;). We drop tags and
    unescape entities so the embedding model reads clean prose.
    """
    text = text.replace("<p>", " ").replace("</p>", " ")
    text = _TAG_RE.sub("", text)
    return html.unescape(text).strip()


def _truncate(text: str, max_chars: int) -> str:
    """Truncate to max_chars at a word boundary."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _fetch_item(item_id: int) -> dict | None:
    """Fetch one item (story or comment) from the Firebase HN API."""
    resp = requests.get(f"{config.HN_FIREBASE_BASE}/item/{item_id}.json", timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_top_comments(story_id: str) -> list[str]:
    """Return up to COMMENTS_PER_STORY of a story's highest-ranked comments.

    The story item's `kids` list is HN's ranked order (best first), so we walk it
    from the front and keep the first few comments that pass the quality filters.
    Top-level only — we never recurse into replies.
    """
    try:
        story = _fetch_item(int(story_id))
    except (requests.RequestException, ValueError):
        return []
    if not story:
        return []

    kids = story.get("kids") or []
    kept: list[str] = []
    for kid_id in kids[: config.COMMENT_SCAN_LIMIT]:
        if len(kept) >= config.COMMENTS_PER_STORY:
            break
        try:
            c = _fetch_item(kid_id)
        except requests.RequestException:
            continue
        if not c or c.get("type") != "comment":
            continue
        if c.get("deleted") or c.get("dead"):  # HN moderation filter
            continue
        text = _strip_html(c.get("text") or "")
        if len(text.split()) < config.COMMENT_MIN_WORDS:  # skip one-liners
            continue
        kept.append(_truncate(text, config.COMMENT_MAX_CHARS))
    return kept


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

    # Second phase: enrich each story with its top-ranked comments (optional).
    # Kept separate from story collection so the request patterns (Algolia vs
    # Firebase) and their progress bars don't tangle.
    if config.COMMENTS_ENABLED:
        for rec in tqdm(records, desc="Fetching top comments", unit="story"):
            rec["top_comments"] = fetch_top_comments(rec["id"])
    else:
        for rec in records:
            rec["top_comments"] = []

    with open(config.ITEMS_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records)


if __name__ == "__main__":
    count = ingest()
    print(f"Wrote {count} stories to {config.ITEMS_PATH}")
