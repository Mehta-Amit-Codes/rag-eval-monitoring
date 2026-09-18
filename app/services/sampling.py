"""
Live sampling. Scoring every single production query would be slow and
costly (it's an extra LLM call per query) -- instead, sample a
configurable percentage. Uses a deterministic hash of the query so the
same query always samples the same way within a namespace, which makes
manual debugging/reproduction easier than pure random sampling.
"""
import hashlib
import os

SAMPLE_RATE = float(os.environ.get("EVAL_SAMPLE_RATE", "0.2"))  # 20% by default


def should_sample(namespace: str, query: str, sample_rate: float = SAMPLE_RATE) -> bool:
    if sample_rate >= 1.0:
        return True
    if sample_rate <= 0.0:
        return False
    digest = hashlib.sha256(f"{namespace}:{query}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF  # -> deterministic float in [0, 1)
    return bucket < sample_rate
