"""Append-only log of radiologist decisions, one row per Approve/Reject.

The viewer is the only writer. Keeping it append-only means a review can be replayed later: what was shown, what
probability it carried, and who decided what — which is the part a regulated version of this pipeline would have to
be able to reconstruct.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_URL = "sqlite:///audit.db"
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    study_uid TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('approved', 'rejected')),
    reason TEXT,
    probability REAL,
    threshold_used REAL,
    model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_study ON decisions (study_uid);
"""


def db_path() -> Path:
    """`sqlite:///audit.db` -> audit.db, `sqlite:////data/audit.db` -> /data/audit.db."""
    url = os.getenv("AUDIT_DB_URL", DEFAULT_URL)
    if url.startswith("sqlite:///"):
        return Path(url[len("sqlite:///") :])
    return Path(url)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def record(
    study_uid: str,
    decision: str,
    reason: str | None = None,
    probability: float | None = None,
    threshold_used: float | None = None,
    model_version: str | None = None,
    path: Path | None = None,
) -> dict:
    if decision not in ("approved", "rejected"):
        raise ValueError(f"decision must be approved or rejected, not {decision!r}")
    target = path or db_path()
    with _lock, _connect(target) as conn:
        cursor = conn.execute(
            "INSERT INTO decisions (recorded_at, study_uid, decision, reason, probability,"
            " threshold_used, model_version) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                study_uid,
                decision,
                reason or None,
                probability,
                threshold_used,
                model_version,
            ),
        )
        row = conn.execute("SELECT * FROM decisions WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)


def for_study(study_uid: str, path: Path | None = None) -> list[dict]:
    target = path or db_path()
    with _lock, _connect(target) as conn:
        rows = conn.execute(
            "SELECT * FROM decisions WHERE study_uid = ? ORDER BY id", (study_uid,)
        ).fetchall()
        return [dict(row) for row in rows]
