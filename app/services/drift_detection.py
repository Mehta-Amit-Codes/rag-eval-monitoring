"""
Retrieval drift detection. Periodically re-runs a fixed set of test
queries against Project 1's retrieval-only endpoint (added in Project 1
specifically for external consumers like this one -- see
multi-tenant-rag/app/routers/query.py: POST /query/retrieve) and compares
the returned chunk IDs to the previous snapshot. A falling overlap score
means the corpus has changed enough that the same question now surfaces
different source material -- worth flagging even if generation quality
looks fine, since it's often the leading indicator of a quality problem.
"""
import os

import requests

from app.core.db import latest_two_snapshots, save_drift_snapshot
from app.services.alerting import check_drift

RAG_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")


def fetch_chunk_ids(api_key: str, test_query: str, top_k: int = 5) -> list[str]:
    resp = requests.post(
        f"{RAG_API_URL}/query/retrieve",
        headers={"X-API-Key": api_key},
        json={"question": test_query, "top_k": top_k},
        timeout=30,
    )
    resp.raise_for_status()
    sources = resp.json()["sources"]
    # document_id + chunk_index uniquely identifies a chunk without needing
    # Project 1 to expose its internal chunk UUID.
    return [f"{s['document_id']}:{s['chunk_index']}" for s in sources]


def _overlap_fraction(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    set_a, set_b = set(a), set(b)
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


def run_drift_check(namespace: str, api_key: str, test_queries: list[str], top_k: int = 5) -> list[dict]:
    """
    For each test query: fetch current top-k chunk IDs from Project 1,
    save as a new snapshot, compare against the previous snapshot (if any),
    and fire an alert if drift exceeds the configured threshold.

    Returns a list of {query, overlap_fraction, drift_alert} per test query.
    """
    results = []
    for query in test_queries:
        current_ids = fetch_chunk_ids(api_key, query, top_k=top_k)
        save_drift_snapshot(namespace, query, current_ids)

        snapshots = latest_two_snapshots(namespace, query)
        if len(snapshots) < 2:
            results.append({"query": query, "overlap_fraction": None, "drift_alert": None})
            continue

        import json
        newest_ids = json.loads(snapshots[0]["chunk_ids"])
        previous_ids = json.loads(snapshots[1]["chunk_ids"])
        overlap = _overlap_fraction(newest_ids, previous_ids)

        alert_msg = check_drift(namespace, query, overlap)
        results.append({"query": query, "overlap_fraction": round(overlap, 3), "drift_alert": alert_msg})

    return results
