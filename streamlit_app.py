"""
Streamlit dashboard for the RAG Evaluation & Monitoring layer.

Tabs:
  - Try it: log a production-style interaction, see if it got sampled + scored
  - Feedback: browse recent scored interactions, give thumbs up/down
  - Drift Check: run fixed test queries against Project 1, see overlap scores
  - Dashboard: rolling score summary, alert history
  - Weekly Report: generate + preview the markdown report on demand

Run:
    streamlit run streamlit_app.py
"""
import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("EVAL_API_URL", "http://localhost:8002")

st.set_page_config(page_title="RAG Eval & Monitoring", page_icon="🔍", layout="wide")
st.title("🔍 RAG Evaluation & Monitoring")
st.caption(f"API: {API_BASE_URL} · Continuous eval + feedback loops + retrieval drift")

namespace = st.sidebar.text_input("Namespace (Project 1 tenant_id)", value="demo-tenant")
project1_api_key = st.sidebar.text_input(
    "Project 1 API key (for drift checks)", type="password",
    help="Only needed for the Drift Check tab — it queries Project 1's /query/retrieve.",
)

tab_try, tab_feedback, tab_drift, tab_dashboard, tab_report = st.tabs(
    ["🧪 Try it", "👍👎 Feedback", "📡 Drift Check", "📊 Dashboard", "📄 Weekly Report"]
)

# ---- Try it -----------------------------------------------------------
with tab_try:
    st.subheader("Log a production-style interaction")
    st.caption(
        "Simulates what your RAG app would call after generating an answer "
        "(e.g. right after Project 1's /query or Project 2's /smart-query response)."
    )

    query = st.text_input("Query", placeholder="What is the refund policy?")
    context = st.text_area("Context (retrieved chunks)", height=100,
                            placeholder="Paste the retrieved chunk text used to generate the answer.")
    answer = st.text_area("Answer", height=100, placeholder="The generated answer to score.")
    force_sample = st.checkbox("Force sampling (bypass sample rate)", value=True,
                                help="Off = respects EVAL_SAMPLE_RATE, so it may skip scoring.")

    if st.button("Log + score", type="primary") and query and answer:
        with st.spinner("Sampling decision + judge scoring..."):
            resp = requests.post(
                f"{API_BASE_URL}/log-interaction",
                json={
                    "namespace": namespace, "query": query, "answer": answer,
                    "context": context, "sources": [], "force_sample": force_sample,
                },
            )
        if resp.ok:
            data = resp.json()
            if not data["sampled"]:
                st.info("Not sampled this time (below sample rate) — no judge call made, $0 cost.")
            else:
                scores = data["scores"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Faithfulness", f"{scores['faithfulness']:.2f}")
                col2.metric("Relevance", f"{scores['relevance']:.2f}")
                col3.metric("Tone", f"{scores['tone']:.2f}")
                st.caption(f"Judge reasoning: {scores['reasoning']}")
                st.caption(f"Interaction ID: `{data['interaction_id']}` — rate it in the Feedback tab.")

                if data["alerts_fired"]:
                    for msg in data["alerts_fired"]:
                        st.warning(f"🚨 {msg}")
        else:
            st.error(f"Failed: {resp.status_code} {resp.text}")

# ---- Feedback ----------------------------------------------------------
with tab_feedback:
    st.subheader("Recent scored interactions")
    limit = st.slider("How many to show", 5, 100, 20)
    resp = requests.get(f"{API_BASE_URL}/interactions/{namespace}", params={"limit": limit})

    if resp.ok:
        interactions = resp.json()
        if not interactions:
            st.info("No scored interactions yet — log some in the Try it tab.")
        for item in interactions:
            with st.container(border=True):
                st.markdown(f"**Q:** {item['query']}")
                st.markdown(f"**A:** {item['answer']}")
                col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
                col1.caption(f"Faithfulness: {item['faithfulness']:.2f}")
                col2.caption(f"Relevance: {item['relevance']:.2f}")
                col3.caption(f"Tone: {item['tone']:.2f}")

                current_feedback = item.get("feedback")
                feedback_label = {1: "👍 marked helpful", -1: "👎 marked unhelpful"}.get(current_feedback)
                if feedback_label:
                    col4.caption(feedback_label)
                else:
                    up_col, down_col = col4.columns(2)
                    if up_col.button("👍", key=f"up_{item['id']}"):
                        requests.post(f"{API_BASE_URL}/feedback", json={
                            "interaction_id": item["id"], "namespace": namespace,
                            "query": item["query"], "answer": item["answer"],
                            "sources": [], "thumbs_up": True,
                        })
                        st.rerun()
                    if down_col.button("👎", key=f"down_{item['id']}"):
                        requests.post(f"{API_BASE_URL}/feedback", json={
                            "interaction_id": item["id"], "namespace": namespace,
                            "query": item["query"], "answer": item["answer"],
                            "sources": [], "thumbs_up": False,
                        })
                        st.info("Added to golden dataset (regression test set).")
                        st.rerun()
    else:
        st.error(f"Failed: {resp.status_code}")

    st.divider()
    st.subheader("Golden dataset (regression test set, from 👎 feedback)")
    gresp = requests.get(f"{API_BASE_URL}/golden-dataset/{namespace}")
    if gresp.ok:
        golden = gresp.json()
        if golden:
            st.dataframe(pd.DataFrame(golden)[["query", "answer", "reason", "added_at"]],
                         use_container_width=True)
        else:
            st.caption("Empty — no negative feedback recorded yet.")

# ---- Drift Check -----------------------------------------------------
with tab_drift:
    st.subheader("Retrieval drift check")
    st.caption(
        "Re-runs a fixed set of canary queries against Project 1's /query/retrieve "
        "and compares retrieved chunks to the last check."
    )
    default_queries = "What is the refund policy?\nHow do I contact support?\nWhat are your business hours?"
    test_queries_raw = st.text_area("Test queries (one per line)", value=default_queries, height=100)

    if st.button("Run drift check", type="primary"):
        if not project1_api_key:
            st.error("Enter Project 1's API key in the sidebar first.")
        else:
            test_queries = [q.strip() for q in test_queries_raw.splitlines() if q.strip()]
            with st.spinner("Fetching current retrieval results and comparing..."):
                resp = requests.post(
                    f"{API_BASE_URL}/drift-check",
                    json={"namespace": namespace, "api_key": project1_api_key,
                          "test_queries": test_queries},
                )
            if resp.ok:
                results = resp.json()
                for r in results:
                    if r["overlap_fraction"] is None:
                        st.info(f"**{r['query']}** — first snapshot recorded, nothing to compare yet.")
                    elif r["drift_alert"]:
                        st.warning(f"🚨 **{r['query']}** — overlap {r['overlap_fraction']:.0%}. {r['drift_alert']}")
                    else:
                        st.success(f"**{r['query']}** — overlap {r['overlap_fraction']:.0%}, stable.")
            else:
                st.error(f"Failed: {resp.status_code} {resp.text}")

# ---- Dashboard --------------------------------------------------------
with tab_dashboard:
    st.subheader("Rolling score summary")
    window_label = st.selectbox("Window", ["Last hour", "Last 24 hours", "Last 7 days"], index=1)
    window_seconds = {"Last hour": 3600, "Last 24 hours": 86400, "Last 7 days": 604800}[window_label]

    resp = requests.get(f"{API_BASE_URL}/stats/{namespace}", params={"window_seconds": window_seconds})
    if resp.ok:
        s = resp.json()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sampled interactions", s.get("n") or 0)
        col2.metric("Avg faithfulness", f"{s['avg_faithfulness']:.2f}" if s.get("avg_faithfulness") else "n/a")
        col3.metric("Avg relevance", f"{s['avg_relevance']:.2f}" if s.get("avg_relevance") else "n/a")
        col4.metric("Avg tone", f"{s['avg_tone']:.2f}" if s.get("avg_tone") else "n/a")

        col5, col6 = st.columns(2)
        col5.metric("👍 Thumbs up", s.get("thumbs_up") or 0)
        col6.metric("👎 Thumbs down", s.get("thumbs_down") or 0)
    else:
        st.error(f"Failed: {resp.status_code}")

    st.divider()
    st.subheader("Alert history")
    aresp = requests.get(f"{API_BASE_URL}/alerts/{namespace}")
    if aresp.ok:
        alerts = aresp.json()
        if alerts:
            st.dataframe(pd.DataFrame(alerts)[["alert_type", "message", "metric_value", "threshold", "timestamp"]],
                         use_container_width=True)
        else:
            st.caption("No alerts fired yet.")

# ---- Weekly Report -----------------------------------------------------
with tab_report:
    st.subheader("Weekly eval report")
    if st.button("Generate report", type="primary"):
        resp = requests.get(f"{API_BASE_URL}/report/{namespace}")
        if resp.ok:
            report_md = resp.json()["report_markdown"]
            st.markdown(report_md)
            st.download_button("Download as .md", report_md, file_name=f"eval_report_{namespace}.md")
        else:
            st.error(f"Failed: {resp.status_code}")
