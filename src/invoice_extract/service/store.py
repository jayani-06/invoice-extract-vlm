"""Persistence: processed documents, the review queue, and reviewer decisions.

SQLite through the standard library, so persistence adds no dependency.

The decisions table is the important one. A corrected vendor name or category
is the only non-synthetic label this project will ever collect, so it is
stored as an **append-only log** -- reviewer, action, field, old value, new
value, reason, timestamp -- and never updated in place. The current state of a
document is derived from that log plus the routing the pipeline chose; nothing
overwrites a past decision, including a later decision on the same document.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    filename        TEXT,
    image_path      TEXT,
    overall_verdict TEXT NOT NULL,
    status          TEXT NOT NULL,
    result_json     TEXT NOT NULL,
    final_json      TEXT
);
CREATE TABLE IF NOT EXISTS review_items (
    item_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id),
    kind        TEXT NOT NULL CHECK (kind IN ('document', 'line_category')),
    line_index  INTEGER,
    state       TEXT NOT NULL CHECK (state IN ('open', 'resolved')),
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id),
    item_id     INTEGER REFERENCES review_items(item_id),
    reviewer    TEXT NOT NULL CHECK (length(trim(reviewer)) > 0),
    action      TEXT NOT NULL,
    field       TEXT,
    old_value   TEXT,
    new_value   TEXT,
    reason      TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_state ON review_items(state);
CREATE INDEX IF NOT EXISTS idx_decisions_doc ON decisions(doc_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReviewStore:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------- documents

    def add_document(
        self,
        *,
        doc_id: str,
        filename: str | None,
        image_path: str | None,
        overall_verdict: str,
        status: str,
        result: dict[str, Any],
        open_items: list[tuple[str, int | None]],
        final_document: dict[str, Any] | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    now(),
                    filename,
                    image_path,
                    overall_verdict,
                    status,
                    json.dumps(result),
                    json.dumps(final_document) if final_document is not None else None,
                ),
            )
            for kind, line_index in open_items:
                conn.execute(
                    "INSERT INTO review_items (doc_id, kind, line_index, state, created_at) "
                    "VALUES (?, ?, ?, 'open', ?)",
                    (doc_id, kind, line_index, now()),
                )

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["result"] = json.loads(out.pop("result_json"))
        final = out.pop("final_json")
        out["final_document"] = json.loads(final) if final else None
        return out

    def set_status(
        self, conn: sqlite3.Connection, doc_id: str, status: str, final: dict | None
    ) -> None:
        conn.execute(
            "UPDATE documents SET status = ?, final_json = ? WHERE doc_id = ?",
            (status, json.dumps(final) if final is not None else None, doc_id),
        )

    def accepted_documents(self) -> list[dict[str, Any]]:
        """Final (reviewer-corrected) documents in acceptance order -- what the
        knowledge base is rebuilt from on start-up."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT d.doc_id, d.final_json FROM documents d "
                "WHERE d.status IN ('auto_accepted', 'accepted', 'accepted_with_override') "
                "AND d.final_json IS NOT NULL "
                "ORDER BY (SELECT MAX(decision_id) FROM decisions x WHERE x.doc_id = d.doc_id), "
                "d.created_at"
            ).fetchall()
        return [json.loads(r["final_json"]) for r in rows]

    # ----------------------------------------------------------------- items

    def items(self, *, state: str | None = "open", doc_id: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM review_items WHERE 1=1", []
        if state:
            sql += " AND state = ?"
            args.append(state)
        if doc_id:
            sql += " AND doc_id = ?"
            args.append(doc_id)
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql + " ORDER BY item_id", args)]

    def get_item(self, item_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM review_items WHERE item_id = ?", (item_id,)
            ).fetchone()
        return dict(row) if row else None

    def resolve_item(self, conn: sqlite3.Connection, item_id: int) -> None:
        conn.execute("UPDATE review_items SET state = 'resolved' WHERE item_id = ?", (item_id,))

    # ------------------------------------------------------------- decisions

    def record(
        self,
        conn: sqlite3.Connection,
        *,
        doc_id: str,
        reviewer: str,
        action: str,
        item_id: int | None = None,
        field: str | None = None,
        old_value: Any = None,
        new_value: Any = None,
        reason: str | None = None,
    ) -> int:
        cur = conn.execute(
            "INSERT INTO decisions (doc_id, item_id, reviewer, action, field, old_value, "
            "new_value, reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                item_id,
                reviewer.strip(),
                action,
                field,
                json.dumps(old_value),
                json.dumps(new_value),
                reason,
                now(),
            ),
        )
        return int(cur.lastrowid)

    def decisions(self, doc_id: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM decisions", []
        if doc_id:
            sql += " WHERE doc_id = ?"
            args.append(doc_id)
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(sql + " ORDER BY decision_id", args)]
        for row in rows:
            row["old_value"] = json.loads(row["old_value"]) if row["old_value"] else None
            row["new_value"] = json.loads(row["new_value"]) if row["new_value"] else None
        return rows
