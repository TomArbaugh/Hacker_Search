"""Layer 4a — LIVE: user accounts backed by SQLite.

Small, self-contained auth store. Passwords are never stored in plaintext —
we keep only a salted hash from werkzeug.security (which ships with Flask, so no
extra dependency). The Flask app (app.py) calls into this module; this module
knows nothing about Flask, mirroring the search_core split.

Table:
    users(id, email UNIQUE, password_hash, created_at)
"""
from __future__ import annotations

import sqlite3

from werkzeug.security import check_password_hash, generate_password_hash

import config

# A real, well-formed hash used as a decoy when the email isn't found, so
# authenticate() always does the same work and can't leak account existence via
# timing. Computed once at import.
_DECOY_HASH = generate_password_hash("decoy-password-never-matched")


def _connect() -> sqlite3.Connection:
    """Open a connection with dict-style row access (row["email"])."""
    conn = sqlite3.connect(config.USERS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the users table if it doesn't exist. Safe to call on every boot."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )


def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def get_user_by_id(user_id: int) -> sqlite3.Row | None:
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return cur.fetchone()


def get_user_by_email(email: str) -> sqlite3.Row | None:
    with _connect() as conn:
        cur = conn.execute(
            "SELECT * FROM users WHERE email = ?", (_normalize_email(email),)
        )
        return cur.fetchone()


def create_user(email: str, password: str) -> sqlite3.Row:
    """Create a new account and return its row.

    Raises ValueError with a user-facing message on bad input or a duplicate
    email, so the signup route can show it directly.
    """
    email = _normalize_email(email)
    if not email:
        raise ValueError("Please enter an email address.")
    if len(password or "") < config.MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"Password must be at least {config.MIN_PASSWORD_LENGTH} characters."
        )
    if get_user_by_email(email) is not None:
        raise ValueError("An account with that email already exists.")

    password_hash = generate_password_hash(password)
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email, password_hash),
        )
        user_id = cur.lastrowid
    return get_user_by_id(user_id)


def authenticate(email: str, password: str) -> sqlite3.Row | None:
    """Return the user row if email+password are valid, else None.

    Always runs a hash check (even when the user is missing) so timing doesn't
    reveal whether an email is registered.
    """
    user = get_user_by_email(email)
    stored_hash = user["password_hash"] if user else _DECOY_HASH
    valid = check_password_hash(stored_hash, password or "")
    return user if (user and valid) else None
