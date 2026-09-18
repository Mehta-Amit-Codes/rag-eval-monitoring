"""
Storage for the eval/monitoring layer. SQLite, matching the observability
pattern from Project 2 -- zero extra infra, queryable directly by the
Streamlit dashboard and easy to inspect with any SQLite browser.

Four tables:
  - scored_interactions: every sampled prod query/answer + judge scores
  - golden_dataset: regression-test set, seeded by user feedback
  - drift_snapshots: retrieved chunk IDs for fixed test queries, over time
  - alerts: history of threshold-breach alerts that were fired
"""
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

DB_PATH = os.environ.get("EVAL_DB", "eval_monitoring.db")


@dataclass
class ScoredInteraction:
    namespace: str
    query: str
    answer: str
    sources: list[dict]
    faithfulness: float
    relevance: float
    tone: float
    judge_reasoning: str
    feedback: int | None = None  # +1 thumbs up, -1 thumbs down, None = no feedback
    timestamp: float = field(default_factory=time.time)


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scored_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT,
                query TEXT,
                answer TEXT,
                sources TEXT,
                faithfulness REAL,
                relevance REAL,
                tone REAL,
                judge_reasoning TEXT,
                feedback INTEGER,
                timestamp REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS golden_dataset (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT,
                query TEXT,
                answer TEXT,
                sources TEXT,
                reason TEXT,          -- why it was added, e.g. "negative feedback"
                added_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS drift_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT,
                test_query TEXT,
                chunk_ids TEXT,       -- JSON list, ordered by rank
                timestamp REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT,
                alert_type TEXT,      -- "score_drop" | "retrieval_drift"
                message TEXT,
                metric_value REAL,
                threshold REAL,
                timestamp REAL
            )
        """)


def insert_interaction(event: ScoredInteraction) -> int:
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO scored_interactions
               (namespace, query, answer, sources, faithfulness, relevance, tone,
                judge_reasoning, feedback, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event.namespace, event.query, event.answer, json.dumps(event.sources),
             event.faithfulness, event.relevance, event.tone, event.judge_reasoning,
             event.feedback, event.timestamp),
        )
        return cur.lastrowid


def set_feedback(interaction_id: int, feedback: int) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE scored_interactions SET feedback = ? WHERE id = ?",
            (feedback, interaction_id),
        )


def add_to_golden_dataset(namespace: str, query: str, answer: str, sources: list[dict], reason: str) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO golden_dataset (namespace, query, answer, sources, reason, added_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (namespace, query, answer, json.dumps(sources), reason, time.time()),
        )


def get_golden_dataset(namespace: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM golden_dataset WHERE namespace = ? ORDER BY added_at DESC",
            (namespace,),
        ).fetchall()
    return [dict(r) for r in rows]


def rolling_score_summary(namespace: str, window_seconds: int) -> dict:
    since = time.time() - window_seconds
    with _conn() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS n,
                      AVG(faithfulness) AS avg_faithfulness,
                      AVG(relevance) AS avg_relevance,
                      AVG(tone) AS avg_tone,
                      SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END) AS thumbs_up,
                      SUM(CASE WHEN feedback = -1 THEN 1 ELSE 0 END) AS thumbs_down
               FROM scored_interactions
               WHERE namespace = ? AND timestamp >= ?""",
            (namespace, since),
        ).fetchone()
    return dict(row) if row else {}


def recent_interactions(namespace: str, limit: int = 100) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            """SELECT * FROM scored_interactions WHERE namespace = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (namespace, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def worst_scoring_interactions(namespace: str, window_seconds: int, limit: int = 10) -> list[dict]:
    since = time.time() - window_seconds
    with _conn() as conn:
        rows = conn.execute(
            """SELECT *, (faithfulness + relevance + tone) / 3.0 AS avg_score
               FROM scored_interactions
               WHERE namespace = ? AND timestamp >= ?
               ORDER BY avg_score ASC LIMIT ?""",
            (namespace, since, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def save_drift_snapshot(namespace: str, test_query: str, chunk_ids: list[str]) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO drift_snapshots (namespace, test_query, chunk_ids, timestamp)
               VALUES (?, ?, ?, ?)""",
            (namespace, test_query, json.dumps(chunk_ids), time.time()),
        )


def latest_two_snapshots(namespace: str, test_query: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            """SELECT * FROM drift_snapshots WHERE namespace = ? AND test_query = ?
               ORDER BY timestamp DESC LIMIT 2""",
            (namespace, test_query),
        ).fetchall()
    return [dict(r) for r in rows]


def log_alert(namespace: str, alert_type: str, message: str, metric_value: float, threshold: float) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO alerts (namespace, alert_type, message, metric_value, threshold, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (namespace, alert_type, message, metric_value, threshold, time.time()),
        )


def recent_alerts(namespace: str, limit: int = 20) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            """SELECT * FROM alerts WHERE namespace = ? ORDER BY timestamp DESC LIMIT ?""",
            (namespace, limit),
        ).fetchall()
    return [dict(r) for r in rows]
