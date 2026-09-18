"""
User feedback loop. Thumbs down on a production answer is a real signal
that offline eval can't generate on its own -- routing it into the golden
dataset means the next time you tune a prompt or swap a model, you're
regression-testing against real failures, not just your original curated
set from Project 3's eval harness.
"""
from app.core.db import add_to_golden_dataset, set_feedback


def record_feedback(interaction_id: int, namespace: str, query: str, answer: str,
                     sources: list[dict], thumbs_up: bool) -> None:
    feedback_value = 1 if thumbs_up else -1
    set_feedback(interaction_id, feedback_value)

    if not thumbs_up:
        # Negative feedback -> becomes a regression test case. Positive
        # feedback isn't added here since a "correct" answer today doesn't
        # need to be pinned as a golden case -- only failures do, to keep
        # the golden set focused on things that broke.
        add_to_golden_dataset(
            namespace=namespace, query=query, answer=answer, sources=sources,
            reason="negative user feedback",
        )
