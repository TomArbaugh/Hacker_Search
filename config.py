"""Single source of truth for settings and paths.

Every other module imports from here, so changing the model, the corpus size,
or where things live on disk is a one-line edit in this file (or an override in
a local .env). Nothing else should hard-code a path or a model name.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load overrides from a local .env if present. Anything here can be overridden
# with an environment variable of the same name.
load_dotenv()

# --- Paths -------------------------------------------------------------------
# Everything generated lives under data/ so it is easy to gitignore and to ship.
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Raw corpus pulled from Hacker News, one JSON object per line.
ITEMS_PATH = DATA_DIR / "hn_items.jsonl"
# Persisted Chroma vector store (the "index").
CHROMA_DIR = DATA_DIR / "chroma"
# SQLite database for user accounts.
USERS_DB_PATH = DATA_DIR / "users.db"

# --- Hacker News ingest (Layer 1) --------------------------------------------
# Algolia's HN Search API: free, no API key, clean JSON. We page backwards
# through time using the created_at_i cursor (see ingest.py).
HN_API_BASE = "http://hn.algolia.com/api/v1"
# How many stories to pull in total. Bigger corpus = better recall but slower
# to embed later. Start small; raise once the pipeline works end to end.
INGEST_TARGET = int(os.getenv("INGEST_TARGET", "2000"))
# Algolia returns at most 1000 hits per page; 1000 keeps requests to a minimum.
INGEST_PAGE_SIZE = int(os.getenv("INGEST_PAGE_SIZE", "1000"))
# Only keep stories with at least this many points — filters out noise so the
# index is full of things people actually found worth reading.
INGEST_MIN_POINTS = int(os.getenv("INGEST_MIN_POINTS", "10"))
# Be polite to the free API: seconds to wait between page requests.
INGEST_REQUEST_DELAY = float(os.getenv("INGEST_REQUEST_DELAY", "0.5"))

# --- Embeddings (Layer 2/3) --------------------------------------------------
# Smallest solid sentence-transformer: 384-dim, fast on CPU, low RAM. Swap for
# "BAAI/bge-small-en-v1.5" later for better quality if the host has headroom.
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Vector store (Layer 2/3) ------------------------------------------------
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "hn_stories")

# --- Search (Layer 3) --------------------------------------------------------
# Default number of results to return from a query.
TOP_K = int(os.getenv("TOP_K", "20"))

# --- Flask / auth (Layer 4) --------------------------------------------------
SECRET_KEY_FILE = BASE_DIR / "secret.key"
# Minimum password length enforced at signup.
MIN_PASSWORD_LENGTH = int(os.getenv("MIN_PASSWORD_LENGTH", "8"))
