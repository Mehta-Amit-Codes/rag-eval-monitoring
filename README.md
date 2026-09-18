# RAG Evaluation & Monitoring

[![Diagram](https://img.shields.io/badge/gitdiagram-view%20architecture-blue)](https://gitdiagram.com/Mehta-Amit-Codes/rag-eval-monitoring)

Reference implementation of the "RAG Evaluation & Monitoring in Production"
blueprint: live sampling, LLM-as-judge scoring, threshold alerting, a
user feedback loop into a golden regression dataset, retrieval drift
detection, and an auto-generated weekly report.

Designed to sit **alongside** Project 1 (Multi-Tenant RAG-as-a-Service)
and Project 2 (RAG Cost Control Layer) — call `/log-interaction` right
after either of those return an answer, and use Project 1's `tenant_id`
as this project's `namespace` throughout.

## How the pieces fit together

```
Project 1 or 2 answers a query
        |
        v
POST /log-interaction  --sampled?--no--> done, $0 extra cost
        | yes
        v
  [LLM-as-judge: faithfulness, relevance, tone]
        |
        v
  [store scored interaction] --> [check rolling-window thresholds] --breach--> [Slack/email alert]

separately, on a schedule:
POST /drift-check  --> re-runs fixed test queries against Project 1's
                        /query/retrieve, compares chunk overlap to last
                        run --> alert if drift exceeds threshold

user clicks 👎 on an answer:
POST /feedback  --> stored, and added to the golden dataset for
                     future regression testing
```

## Run it

```bash
cp .env.example .env
# then edit .env and fill in your real GROK_API_KEY

pip install -r requirements.txt
uvicorn app.main:app --port 8002 --reload
```

In a second terminal:
```bash
export EVAL_API_URL=http://localhost:8002
streamlit run streamlit_app.py
```

## Demo flow

1. **Try it** tab — log a query/context/answer (e.g. copy a real response
   from Project 1's `/query`). With "Force sampling" checked, it always
   scores; unchecked, it respects `EVAL_SAMPLE_RATE` (default 20%).
2. **Feedback** tab — thumbs up/down on the logged interaction. A 👎 adds
   it to the golden dataset shown below.
3. **Drift Check** tab — paste Project 1's tenant API key, run the default
   canary queries. First run just records a baseline (nothing to compare
   yet); run it again after ingesting/changing documents in Project 1 to
   see the overlap score drop and an alert fire if it crosses the
   threshold.
4. **Dashboard** tab — rolling score averages and alert history.
5. **Weekly Report** tab — generates the markdown report on demand
   (stretch goal: wire this to a cron job / GitHub Action that posts it
   to Slack or commits it to a `reports/` folder weekly).

## API

- `POST /log-interaction` — sample + score a production Q&A pair.
- `POST /feedback` — record thumbs up/down; negative feedback → golden dataset.
- `GET /golden-dataset/{namespace}` — the regression test set so far.
- `POST /drift-check` — re-run canary queries against Project 1, compare to last snapshot.
- `GET /stats/{namespace}?window_seconds=86400` — rolling score summary.
- `GET /interactions/{namespace}?limit=100` — recent scored interactions.
- `GET /alerts/{namespace}` — alert history.
- `GET /report/{namespace}` — generates the weekly markdown report.

## Integration with Projects 1 & 2

- **Logging:** after Project 1's `/query` or Project 2's `/smart-query`
  returns, call this project's `/log-interaction` with the same
  `namespace` (Project 1's `tenant_id`), the question, the answer, and
  the retrieved context/sources. This is a fire-and-forget call from the
  caller's perspective — sampling means most calls skip the judge
  entirely.
- **Drift checks:** uses Project 1's `POST /query/retrieve` endpoint
  (the retrieval-only endpoint added for Project 2's integration) so
  drift checks never trigger Project 1's own LLM generation.
- **Cost correlation:** because `namespace` matches across all three
  projects, you can line up Project 2's `/stats/{namespace}` (cost,
  cache hit rate) next to this project's `/stats/{namespace}` (quality
  scores) to see whether a cost-saving change (e.g. routing more queries
  to the cheap model) came with a quality trade-off.

## Notes on the illustrative pieces

- The LLM-as-judge in `llm_judge.py` is a custom rubric rather than
  RAGAS/DeepEval, to keep the dependency footprint light — the module's
  docstring shows exactly where to plug RAGAS in if you want its metric
  suite instead.
- Sampling is deterministic per query (hash-based), not purely random, so
  the same query always samples the same way within a namespace — makes
  a flagged interaction reproducible.
- Retrieval drift is measured via Jaccard overlap of chunk IDs
  (`document_id:chunk_index`) between the two most recent snapshots per
  test query — simple and cheap, no embedding comparison needed.

## Project layout

```
app/
  core/
    db.py                 # SQLite: scored interactions, golden dataset, drift snapshots, alerts
  services/
    sampling.py             # deterministic hash-based sampling decision
    llm_judge.py              # Grok-based LLM-as-judge (faithfulness/relevance/tone)
    alerting.py                # Slack + email, threshold checks for scores and drift
    drift_detection.py           # re-runs canary queries against Project 1, computes overlap
    feedback.py                   # thumbs up/down -> golden dataset
    orchestrator.py                 # ties sampling + judging + storage + alerting together
    report.py                        # weekly markdown report generator
  main.py                             # FastAPI: log-interaction, feedback, drift-check, stats, report
streamlit_app.py                       # dashboard: try-it, feedback review, drift check, report
```
