"""
Orchestrates a single production interaction through the eval pipeline:
sample -> score -> store -> check rolling thresholds -> alert if breached.

Called from the /log-interaction endpoint. The caller supplies the
context/question/answer it already has (e.g. from a Project 1 /query
response) -- this module doesn't call the RAG pipeline itself, since
sampling should happen at the point where a real answer was already
generated, not as a separate synthetic call.
"""
import os

from app.core.db import ScoredInteraction, insert_interaction, rolling_score_summary
from app.services.alerting import check_score_thresholds
from app.services.llm_judge import score_interaction
from app.services.sampling import should_sample

ALERT_WINDOW_SECONDS = int(os.environ.get("ALERT_WINDOW_SECONDS", str(60 * 60 * 24)))  # 24h
MIN_SAMPLES_BEFORE_ALERTING = int(os.environ.get("MIN_SAMPLES_BEFORE_ALERTING", "10"))


def log_interaction(namespace: str, query: str, answer: str, context: str,
                     sources: list[dict], force_sample: bool = False) -> dict:
    """
    Returns {"sampled": bool, "scores": dict | None, "alerts_fired": list[str]}.
    If not sampled, scores is None and no judge call is made (saves cost).
    """
    if not force_sample and not should_sample(namespace, query):
        return {"sampled": False, "scores": None, "alerts_fired": []}

    scores = score_interaction(context, query, answer)

    interaction_id = insert_interaction(ScoredInteraction(
        namespace=namespace, query=query, answer=answer, sources=sources,
        faithfulness=scores["faithfulness"], relevance=scores["relevance"],
        tone=scores["tone"], judge_reasoning=scores["reasoning"],
    ))

    alerts_fired = []
    summary = rolling_score_summary(namespace, ALERT_WINDOW_SECONDS)
    if summary.get("n", 0) >= MIN_SAMPLES_BEFORE_ALERTING:
        alerts_fired = check_score_thresholds(namespace, summary)

    return {
        "sampled": True,
        "interaction_id": interaction_id,
        "scores": scores,
        "alerts_fired": alerts_fired,
    }
