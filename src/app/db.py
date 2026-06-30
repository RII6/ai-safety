"""app/db.py — PostgreSQL store for scan reports.

Replaces the reports/*.json file cache with a `scans` table: the full report is
kept as JSONB and a few fields are extracted into columns for fast listing,
history and filtering. ``cache_key`` (the content hash from scan.py) is UNIQUE
and serves as the cache lookup.

Connection string comes from DATABASE_URL (see config.py); a running PostgreSQL
is required. Schema is created on demand via init_db().
"""
from __future__ import annotations

import psycopg
from psycopg.types.json import Jsonb

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id                BIGSERIAL   PRIMARY KEY,
    repo              TEXT        NOT NULL,
    cache_key         TEXT        NOT NULL UNIQUE,
    verdict_code      TEXT        NOT NULL,
    represents_harm   BOOLEAN,
    params            BIGINT,
    weight_bytes      BIGINT,
    sample            INTEGER,
    dtype             TEXT,
    generated_harmful INTEGER     NOT NULL DEFAULT 0,
    generated_benign  INTEGER     NOT NULL DEFAULT 0,
    elapsed_s         DOUBLE PRECISION,
    report            JSONB       NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_scans_repo ON scans (repo);
CREATE INDEX IF NOT EXISTS idx_scans_created ON scans (created_at DESC);
"""


def _connect():
    return psycopg.connect(config.DATABASE_URL)


def init_db() -> None:
    """Create the scans table and indexes if absent. Call once at startup."""
    with _connect() as conn:
        conn.execute(_SCHEMA)


def get_cached(cache_key: str) -> dict | None:
    """Return the stored report dict for *cache_key*, or None on a miss."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT report FROM scans WHERE cache_key = %s", (cache_key,)
        ).fetchone()
    return row[0] if row else None


def save_scan(repo: str, cache_key: str, result: dict) -> None:
    """Persist a freshly computed report; upsert on cache_key."""
    meta = result.get("meta", {})
    verdict = result.get("verdict", {})
    gen = meta.get("generated", {})
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO scans (repo, cache_key, verdict_code, represents_harm,
                               params, weight_bytes, sample, dtype,
                               generated_harmful, generated_benign, elapsed_s, report)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (cache_key) DO UPDATE
                SET report = EXCLUDED.report,
                    verdict_code = EXCLUDED.verdict_code,
                    created_at = now()
            """,
            (
                repo, cache_key, verdict.get("code"), verdict.get("represents_harm"),
                meta.get("params"), meta.get("weight_bytes"), meta.get("sample"),
                meta.get("dtype"), gen.get("harmful", 0), gen.get("benign", 0),
                meta.get("elapsed_s"), Jsonb(result),
            ),
        )


def list_scans() -> list[dict]:
    """Most-recent-first list for the reports view: [{id, repo, verdict}, ...]."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, repo, verdict_code FROM scans ORDER BY created_at DESC"
        ).fetchall()
    return [{"id": r[0], "repo": r[1], "verdict": r[2]} for r in rows]

def get_scan_by_id(scan_id: int) -> dict | None:
    """Return the full report for a given scan ID, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT report FROM scans WHERE id = %s", (scan_id,)
        ).fetchone()
    return row[0] if row else None
