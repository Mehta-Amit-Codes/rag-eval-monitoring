"""
LLM-as-judge scoring. Deliberately dependency-light (no RAGAS/DeepEval
import here) so the judge call is transparent and easy to swap models on
-- but the rubric below mirrors what RAGAS calls "faithfulness" and
"answer relevancy", plus a tone check, and the docstring notes where
you'd plug RAGAS/DeepEval in instead if you want their metric suite.
"""
import json
import os
import re

from openai import OpenAI

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "grok-4")
GROK_BASE_URL = os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1")

JUDGE_SYSTEM_PROMPT = """You are a strict, consistent evaluator of RAG (retrieval-augmented \
generation) answers. You will be given retrieved context, a question, and an answer. Score the \
answer on three dimensions, each from 0.0 to 1.0:

- faithfulness: Is every claim in the answer actually supported by the provided context? \
1.0 = fully grounded, 0.0 = fabricated / contradicts context.
- relevance: Does the answer actually address the question asked? \
1.0 = directly on-point, 0.0 = off-topic or non-answer.
- tone: Is the answer clear, appropriately hedged when context is thin, and professional? \
1.0 = excellent tone, 0.0 = poor (overconfident, rude, confusing).

Respond with ONLY a JSON object, no other text:
{"faithfulness": <float>, "relevance": <float>, "tone": <float>, "reasoning": "<one or two sentences>"}
"""


def score_interaction(context: str, question: str, answer: str) -> dict:
    """
    Returns {"faithfulness": float, "relevance": float, "tone": float, "reasoning": str}.
    Falls back to a defensive zero-score result (with the parse error surfaced
    in `reasoning`) if the judge doesn't return valid JSON, so a malformed
    judge response never crashes the sampling pipeline.
    """
    client = OpenAI(api_key=os.environ["GROK_API_KEY"], base_url=GROK_BASE_URL)

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {answer}"

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        temperature=0,  # judge should be as deterministic as possible
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content or ""

    try:
        # Judges occasionally wrap JSON in ```json fences despite instructions -- strip them.
        cleaned = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(cleaned)
        return {
            "faithfulness": float(parsed["faithfulness"]),
            "relevance": float(parsed["relevance"]),
            "tone": float(parsed["tone"]),
            "reasoning": parsed.get("reasoning", ""),
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return {
            "faithfulness": 0.0,
            "relevance": 0.0,
            "tone": 0.0,
            "reasoning": f"[judge parse error: {e}] raw response: {raw[:200]}",
        }


# --- RAGAS/DeepEval integration point -----------------------------------
# To use RAGAS's own metric suite instead of (or alongside) the custom
# judge above:
#
#   from ragas import evaluate
#   from ragas.metrics import faithfulness, answer_relevancy
#   from datasets import Dataset
#
#   dataset = Dataset.from_dict({
#       "question": [question], "answer": [answer], "contexts": [[context]],
#   })
#   result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
#
# RAGAS's own LLM call needs the same GROK_API_KEY wired through its LLM
# wrapper (langchain-style) -- left out of the default path here to avoid
# pulling in RAGAS's heavier dependency tree for a project meant to run
# with a light footprint. See RAGAS docs for the LangchainLLMWrapper setup.
