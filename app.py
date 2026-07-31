"""Layer 4b — LIVE: the public Flask web app.

One page: a search box and results. No accounts, no login wall — the whole app
is a thin shell over search_core.search(). Importing search_core loads the model
and opens the Chroma index once at startup (see that module), so the web layer
here stays tiny.

Run locally (dev):
    flask --app app run --debug

Run in production (Docker / HF Spaces):
    gunicorn --bind 0.0.0.0:7860 app:app
"""
import logging
import os
import sys
import traceback

from flask import Flask, render_template, request

from search_core import search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/")
def index():
    """The search page. Query comes in via ?q= so results are shareable URLs."""
    logger.info("=== REQUEST RECEIVED ===")
    logger.info(f"Full URL: {request.url}")
    logger.info(f"Args: {dict(request.args)}")
    
    query = request.args.get("q", "").strip()
    logger.info(f"Query extracted: '{query}'")
    
    try:
        logger.info("Calling search()...")
        results = search(query)
        logger.info(f"Search returned {len(results)} results")
    except Exception as e:
        logger.error(f"Search FAILED with exception: {e}")
        logger.error(traceback.format_exc())
        results = []
    
    logger.info("Rendering template...")
    return render_template("search.html", query=query, results=results)


@app.route("/healthz")
def healthz():
    """Lightweight liveness endpoint for the uptime pinger that keeps the free
    Space warm. Does no model work, so pinging it is cheap."""
    return "ok", 200


if __name__ == "__main__":
    # Local/dev entrypoint. In production, gunicorn imports `app` directly and
    # this block never runs. Host/port/debug are environment-driven so the same
    # file works locally and on a host that injects $PORT.
    port = int(os.getenv("PORT", "7860"))
    debug = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
