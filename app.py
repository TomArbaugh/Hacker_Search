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

# Set up logging FIRST, before any other imports that might fail
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

print("=== APP.PY STARTING ===", flush=True)
logger.info("=== APP.PY STARTING ===")

from flask import Flask, render_template, request

print("=== FLASK IMPORTED OK ===", flush=True)
logger.info("=== FLASK IMPORTED OK ===")

# Import search_core with error handling - this is where it might fail
search = None
try:
    print("=== IMPORTING SEARCH_CORE ===", flush=True)
    logger.info("=== IMPORTING SEARCH_CORE ===")
    from search_core import search
    print("=== SEARCH_CORE IMPORTED OK ===", flush=True)
    logger.info("=== SEARCH_CORE IMPORTED OK ===")
except Exception as e:
    print(f"=== SEARCH_CORE IMPORT FAILED: {e} ===", flush=True)
    logger.error(f"=== SEARCH_CORE IMPORT FAILED: {e} ===")
    logger.error(traceback.format_exc())
    print(traceback.format_exc(), flush=True)

print("=== CREATING FLASK APP ===", flush=True)
logger.info("=== CREATING FLASK APP ===")

app = Flask(__name__)

logger.info("=== FLASK APP CREATED ===")
print("=== FLASK APP CREATED (print) ===", flush=True)


@app.before_request
def log_request():
    """Log every incoming request before processing."""
    print(f"=== BEFORE_REQUEST: {request.method} {request.url} ===", flush=True)
    logger.info(f"=== BEFORE_REQUEST: {request.method} {request.url} ===")


@app.route("/")
def index():
    """The search page. Query comes in via ?q= so results are shareable URLs."""
    logger.info("=== REQUEST RECEIVED ===")
    logger.info(f"Full URL: {request.url}")
    logger.info(f"Args: {dict(request.args)}")
    
    query = request.args.get("q", "").strip()
    logger.info(f"Query extracted: '{query}'")
    
    if search is None:
        logger.error("search_core not loaded - returning empty results")
        return "<h1>Error</h1><p>Search module failed to load. Check container logs.</p>", 500
    
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


@app.route("/test")
def test_route():
    """Debug route to test if query parameters work at all."""
    print("=== TEST ROUTE HIT ===", flush=True)
    logger.info("=== TEST ROUTE HIT ===")
    q = request.args.get("q", "no-q-param")
    msg = f"Test route works! q={q}, all args={dict(request.args)}"
    logger.info(msg)
    print(msg, flush=True)
    return f"<h1>Test Route</h1><p>{msg}</p>", 200


if __name__ == "__main__":
    # Local/dev entrypoint. In production, gunicorn imports `app` directly and
    # this block never runs. Host/port/debug are environment-driven so the same
    # file works locally and on a host that injects $PORT.
    port = int(os.getenv("PORT", "7860"))
    debug = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
