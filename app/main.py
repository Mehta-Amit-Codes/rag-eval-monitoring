from dotenv import load_dotenv
load_dotenv()  # loads .env for local `uvicorn` runs; no-op under Docker Compose

from fastapi import FastAPI
from pydantic import BaseModel

from app.core.db import get_golden_dataset, init_db, recent_alerts, recent_interactions, rolling_score_summary
from app.services.drift_detection import run_drift_check
from app.services.feedback import record_feedback
from app.services.orchestrator import log_interaction
from app.services.report import generate_weekly_report

app = FastAPI(
    title="RAG Evaluation & Monitoring",
    description="Continuous eval, feedback loops, and retrieval-drift detection "
                "for a deployed RAG app (built to sit alongside Project 1 / Project 2).",
    version="0.1.0",
)

init_db()


# ---- Log + score a production interaction --------------------------------
class Source(BaseModel):
    document_id: str
    chunk_index: int
    text: str


class LogInteractionRequest(BaseModel):
    namespace: str          # Project 1 tenant_id (or any app-defined scope)
    query: str
    answer: str
    context: str            # concatenated retrieved chunk text, for the judge
    sources: list[Source] = []
    force_sample: bool = False


@app.post("/log-interaction")
def log_interaction_endpoint(req: LogInteractionRequest):
    return log_interaction(
        namespace=req.namespace,
        query=req.query,
        answer=req.answer,
        context=req.context,
        sources=[s.model_dump() for s in req.sources],
        force_sample=req.force_sample,
    )


# ---- Feedback --------------------------------------------------------
class FeedbackRequest(BaseModel):
    interaction_id: int
    namespace: str
    query: str
    answer: str
    sources: list[Source] = []
    thumbs_up: bool


@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest):
    record_feedback(
        interaction_id=req.interaction_id, namespace=req.namespace,
        query=req.query, answer=req.answer,
        sources=[s.model_dump() for s in req.sources], thumbs_up=req.thumbs_up,
    )
    return {"status": "ok"}


@app.get("/golden-dataset/{namespace}")
def golden_dataset_endpoint(namespace: str):
    return get_golden_dataset(namespace)


# ---- Retrieval drift ------------------------------------------------
class DriftCheckRequest(BaseModel):
    namespace: str
    api_key: str             # Project 1 tenant API key
    test_queries: list[str]  # fixed set of canary queries
    top_k: int = 5


@app.post("/drift-check")
def drift_check_endpoint(req: DriftCheckRequest):
    return run_drift_check(req.namespace, req.api_key, req.test_queries, top_k=req.top_k)


# ---- Stats / dashboard data ------------------------------------------
@app.get("/stats/{namespace}")
def stats_endpoint(namespace: str, window_seconds: int = 86400):
    return rolling_score_summary(namespace, window_seconds)


@app.get("/interactions/{namespace}")
def interactions_endpoint(namespace: str, limit: int = 100):
    return recent_interactions(namespace, limit=limit)


@app.get("/alerts/{namespace}")
def alerts_endpoint(namespace: str, limit: int = 20):
    return recent_alerts(namespace, limit=limit)


# ---- Weekly report (stretch goal) -------------------------------------
@app.get("/report/{namespace}", response_class=None)
def report_endpoint(namespace: str):
    return {"report_markdown": generate_weekly_report(namespace)}


@app.get("/health")
def health():
    return {"status": "ok"}
