import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests
import pandas as pd

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="LLM Eval Leaderboard",
    page_icon="🏆",
    layout="wide",
)


# ── data fetching ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)  # cache for 30 seconds — don't hammer the API
def fetch_runs() -> list[dict]:
    response = requests.get(f"{API_BASE}/runs")
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30)
def fetch_leaderboard(run_id: int) -> pd.DataFrame:
    response = requests.get(f"{API_BASE}/runs/{run_id}/leaderboard")
    response.raise_for_status()
    return pd.DataFrame(response.json())


@st.cache_data(ttl=30)
def fetch_results(run_id: int) -> pd.DataFrame:
    response = requests.get(f"{API_BASE}/runs/{run_id}/results")
    response.raise_for_status()
    return pd.DataFrame(response.json())


# ── sidebar ─────────────────────────────────────────────────────────────────────

st.sidebar.title("LLM Eval Framework")
st.sidebar.caption("Production benchmark results")

try:
    runs = fetch_runs()
    run_options = {f"Run {r['id']}: {r['run_name']}": r["id"] for r in runs}
    selected_run_name = st.sidebar.selectbox("Select Run", list(run_options.keys()))
    selected_run_id = run_options[selected_run_name]

    selected_run = next(r for r in runs if r["id"] == selected_run_id)
    st.sidebar.metric("Models", selected_run["num_models"])
    st.sidebar.metric("Samples", selected_run["num_samples"])
    if selected_run["total_cost"]:
        st.sidebar.metric("Total Cost", f"${selected_run['total_cost']:.4f}")

except requests.exceptions.ConnectionError:
    st.error("Cannot connect to API. Is the server running?")
    st.code("uvicorn src.api.main:app --reload --port 8000")
    st.stop()


# ── main content ────────────────────────────────────────────────────────────────

st.title("🏆 LLM Evaluation Leaderboard")

leaderboard_df = fetch_leaderboard(selected_run_id)

if leaderboard_df.empty:
    st.warning("No results for this run yet.")
    st.stop()


# ── tab layout ──────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Leaderboard", "Radar Chart", "Raw Results"])


with tab1:
    st.subheader("Average Scores by Model and Metric")

    # pivot: rows=models, columns=metrics
    pivot = leaderboard_df.pivot_table(
        index="model_name",
        columns="metric_name",
        values="avg_score",
        aggfunc="mean",
    ).round(4)

    # color the cells — green=high, red=low
    st.dataframe(
        pivot.style.background_gradient(cmap="RdYlGn", axis=None),
        width='stretch',
    )

    st.caption("Scores normalized 0.0–1.0. Green = better.")

    # bar chart for factual accuracy specifically
    if "factual_accuracy" in leaderboard_df["metric_name"].values:
        accuracy_df = leaderboard_df[
            leaderboard_df["metric_name"] == "factual_accuracy"
        ].sort_values("avg_score", ascending=False)

        fig = px.bar(
            accuracy_df,
            x="model_name",
            y="avg_score",
            color="avg_score",
            color_continuous_scale="RdYlGn",
            title="Factual Accuracy by Model",
            labels={"avg_score": "Score", "model_name": "Model"},
            range_y=[0, 1],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')


with tab2:
    st.subheader("Model Capability Radar")
    st.caption("Each axis is a metric. Larger area = stronger overall performance.")

    metrics = leaderboard_df["metric_name"].unique().tolist()
    models = leaderboard_df["model_name"].unique().tolist()

    fig = go.Figure()

    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

    for i, model in enumerate(models):
        model_data = leaderboard_df[leaderboard_df["model_name"] == model]
        scores = []
        for metric in metrics:
            row = model_data[model_data["metric_name"] == metric]
            scores.append(float(row["avg_score"].values[0]) if len(row) > 0 else 0)

        # close the radar polygon
        scores_closed = scores + [scores[0]]
        metrics_closed = metrics + [metrics[0]]

        fig.add_trace(go.Scatterpolar(
            r=scores_closed,
            theta=metrics_closed,
            fill="toself",
            name=model,
            line_color=colors[i % len(colors)],
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=500,
    )
    st.plotly_chart(fig, width='stretch')


with tab3:
    st.subheader("Raw Model Responses")

    results_df = fetch_results(selected_run_id)

    model_filter = st.selectbox(
        "Filter by model",
        ["All"] + sorted(results_df["model_name"].unique().tolist())
    )

    if model_filter != "All":
        results_df = results_df[results_df["model_name"] == model_filter]

    # show key columns only
    display_cols = ["model_name", "dataset", "latency_ms", "cost_usd"]
    st.dataframe(results_df[display_cols], width='stretch')

    # click a row to see the full prompt/response
    st.caption("Expand a row to see the full prompt and response.")
    if st.checkbox("Show full prompts and responses"):
        for _, row in results_df.iterrows():
            with st.expander(f"{row['model_name']} — {row.get('dataset', '')}"):
                st.text_area("Prompt", row["prompt"], height=100)
                st.text_area("Response", row["response"], height=80)
                st.json(row.get("metric_scores", {}))