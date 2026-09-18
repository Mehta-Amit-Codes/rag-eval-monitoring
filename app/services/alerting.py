"""
Alerting: fires when the rolling-window average of any judge metric drops
below a configured threshold, or when retrieval drift crosses its own
threshold. Two channels, either or both can be configured via env vars --
if neither is set, alerts are still logged to the DB (see core/db.py) so
the dashboard shows them even without external notification wired up.
"""
import json
import os
import smtplib
from email.mime.text import MIMEText

import requests

from app.core.db import log_alert

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
ALERT_EMAIL_TO = os.environ.get("ALERT_EMAIL_TO", "")
ALERT_EMAIL_FROM = os.environ.get("ALERT_EMAIL_FROM", "")
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")

FAITHFULNESS_THRESHOLD = float(os.environ.get("FAITHFULNESS_THRESHOLD", "0.7"))
RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "0.7"))
TONE_THRESHOLD = float(os.environ.get("TONE_THRESHOLD", "0.7"))
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.5"))  # fraction of top-k changed


def check_score_thresholds(namespace: str, summary: dict) -> list[str]:
    """Compares a rolling-window summary (from db.rolling_score_summary) against
    thresholds and fires an alert per breached metric. Returns fired alert messages."""
    fired = []
    checks = [
        ("faithfulness", summary.get("avg_faithfulness"), FAITHFULNESS_THRESHOLD),
        ("relevance", summary.get("avg_relevance"), RELEVANCE_THRESHOLD),
        ("tone", summary.get("avg_tone"), TONE_THRESHOLD),
    ]
    for metric_name, value, threshold in checks:
        if value is not None and value < threshold:
            msg = (
                f"[{namespace}] Rolling average {metric_name} = {value:.2f}, "
                f"below threshold {threshold:.2f} (n={summary.get('n', 0)} samples)"
            )
            fired.append(msg)
            log_alert(namespace, "score_drop", msg, value, threshold)
            _send(msg)
    return fired


def check_drift(namespace: str, test_query: str, overlap_fraction: float) -> str | None:
    """overlap_fraction: fraction of chunk IDs shared between the two most recent
    snapshots for this test query. Low overlap = high drift."""
    changed_fraction = 1.0 - overlap_fraction
    if changed_fraction > DRIFT_THRESHOLD:
        msg = (
            f"[{namespace}] Retrieval drift for test query \"{test_query}\": "
            f"{changed_fraction:.0%} of top results changed since last check "
            f"(threshold {DRIFT_THRESHOLD:.0%})"
        )
        log_alert(namespace, "retrieval_drift", msg, changed_fraction, DRIFT_THRESHOLD)
        _send(msg)
        return msg
    return None


def _send(message: str) -> None:
    if SLACK_WEBHOOK_URL:
        try:
            requests.post(SLACK_WEBHOOK_URL, data=json.dumps({"text": message}),
                           headers={"Content-Type": "application/json"}, timeout=10)
        except requests.RequestException:
            pass  # don't let a Slack outage break the scoring pipeline

    if ALERT_EMAIL_TO and SMTP_HOST:
        try:
            msg = MIMEText(message)
            msg["Subject"] = "RAG Eval Alert"
            msg["From"] = ALERT_EMAIL_FROM or SMTP_USER
            msg["To"] = ALERT_EMAIL_TO
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls()
                if SMTP_USER:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(msg["From"], [ALERT_EMAIL_TO], msg.as_string())
        except Exception:
            pass  # same reasoning -- alerting failures shouldn't cascade
