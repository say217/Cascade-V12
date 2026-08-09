import sqlite3
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# SQLite storage
# ---------------------------------------------------------------------------
# Project root = two levels up from this file (app2/db.py -> app2 -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "cascade_db"
DB_PATH = DB_DIR / "cascade.db"


def get_db_connection() -> sqlite3.Connection:
    """Create the cascade_db/ folder + cascade.db file on first use, then connect."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_tables() -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                is_verified INTEGER NOT NULL DEFAULT 0,
                verification_code TEXT,
                code_expires_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# User queries — kept here so routes.py just calls plain functions and never
# touches SQL directly.
# ---------------------------------------------------------------------------
def find_user_by_email_or_username(email: str, username: str) -> sqlite3.Row | None:
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT id FROM users WHERE email = ? OR username = ?",
            (email, username),
        ).fetchone()
    finally:
        conn.close()


def create_user(
    email: str,
    username: str,
    password_hash: str,
    verification_code: str,
    expires_at: str,
) -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO users (email, username, password_hash, is_verified, verification_code, code_expires_at)
            VALUES (?, ?, ?, 0, ?, ?)
            """,
            (email, username, password_hash, verification_code, expires_at),
        )
        conn.commit()
    finally:
        conn.close()


def get_user_for_verification(email: str) -> sqlite3.Row | None:
    conn = get_db_connection()
    try:
        return conn.execute(
            """
            SELECT id, code_expires_at, is_verified, verification_code
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()
    finally:
        conn.close()


def mark_user_verified(user_id: int) -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            """
            UPDATE users
            SET is_verified = 1, verification_code = NULL, code_expires_at = NULL
            WHERE id = ?
            """,
            (user_id,),
        )
        conn.commit()
    finally:
        conn.close()


def get_user_for_resend(email: str) -> sqlite3.Row | None:
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT id, is_verified FROM users WHERE email = ?", (email,)
        ).fetchone()
    finally:
        conn.close()


def set_new_verification_code(user_id: int, verification_code: str, expires_at: str) -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            "UPDATE users SET verification_code = ?, code_expires_at = ? WHERE id = ?",
            (verification_code, expires_at, user_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_user_for_login(email: str) -> sqlite3.Row | None:
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT id, password_hash, is_verified FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    finally:
        conn.close()


def is_code_expired(code_expires_at: str | None) -> bool:
    return bool(code_expires_at) and datetime.fromisoformat(code_expires_at) < datetime.utcnow()