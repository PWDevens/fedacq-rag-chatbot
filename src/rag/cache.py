"""
Persistent exact-match answer cache (SQLite).

Keyed by sha256(normalize(question) | RAG_MODE | RERANK) so different retrieval
strategies cache separately. Survives restarts, so a repeated question replays
instantly instead of re-running retrieval + generation. Per-machine and
gitignored (data/cache.db); not committed, to avoid serving stale answers.
"""

import hashlib
import json
import os
import sqlite3
import threading

from rag.config import RagConfig

_lock = threading.Lock()
_conn = None


def enabled():
    return RagConfig.ANSWER_CACHE


def _connect():
    global _conn
    if _conn is None:
        path = RagConfig.ANSWER_CACHE_PATH
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _conn = sqlite3.connect(path, check_same_thread=False)
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS answers "
            "(key TEXT PRIMARY KEY, answer TEXT, citations TEXT)"
        )
        _conn.commit()
    return _conn


def _key(question):
    norm = " ".join((question or "").lower().split())
    raw = f"{norm}|{RagConfig.RAG_MODE}|{RagConfig.RERANK}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get(question):
    """Return {'answer', 'citations'} for a cached question, or None on miss."""
    if not enabled():
        return None
    with _lock:
        row = _connect().execute(
            "SELECT answer, citations FROM answers WHERE key = ?", (_key(question),)
        ).fetchone()
    if not row:
        return None
    return {"answer": row[0], "citations": json.loads(row[1] or "[]")}


def put(question, answer, citations):
    """Store an answer + citations for a question (no-op when disabled)."""
    if not enabled():
        return
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT OR REPLACE INTO answers (key, answer, citations) VALUES (?, ?, ?)",
            (_key(question), answer, json.dumps(citations or [])),
        )
        conn.commit()
