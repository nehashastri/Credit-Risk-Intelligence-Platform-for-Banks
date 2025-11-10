# streamlit/pages/ml_lifecycle.py
# ML Lifecycle & Model Registry  — Credit Risk Intelligence Platform
# Tabs:
#   1) Overview / Model Registry
#   2) Model Performance Explorer
#   3) Metric Trends & Drift Analysis
#   4) Deployment & Version History

import os
from datetime import datetime
import json
import math
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# --- BigQuery ---
from google.cloud import bigquery

# =========================
# CONFIG (adjust if needed)
# =========================
PROJECT_ID = os.getenv("GCP_PROJECT") or "pipeline-882-team-project"
DATASET_MLOPS = "mlops"   # change if your schema name differs
TABLE_TRAINING_RUN = f"`{PROJECT_ID}.{DATASET_MLOPS}.training_run`"
TABLE_MODEL_VERSION = f"`{PROJECT_ID}.{DATASET_MLOPS}.model_version`"
TABLE_DEPLOYMENT = f"`{PROJECT_ID}.{DATASET_MLOPS}.deployment`"

# Optional: table of scored predictions (for Actual vs Predicted)
# Expected columns: year, week, delinq, delinq_predicted, model_name (or model_version_id)
PRED_TABLE = f"`{PROJECT_ID}.gold.fact_delinquency_predictions`"

# Supported models (for dropdowns / filters)
MODEL_NAMES = [
    "base",               # rolling mean baseline
    "linear_regression",
    "elastic_net",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
]

# Metrics we expect in JSON (string keys in JSON inside tables)
# JSON paths: $.sMAPE, $.RMSE_recent6, $.MAE, $.Pearson_r, $.R2, $.MASE
METRIC_KEYS = ["sMAPE", "RMSE_recent6", "MAE", "Pearson_r", "R2", "MASE"]


# ============
# PAGE SETUP
# ============
st.set_page_config(
    page_title="🤖 ML Lifecycle & Model Registry",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<h1 style='text-align:center;margin-top:0'>🤖 ML Lifecycle & Model Registry</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Track model performance, versioning, and deployment status for delinquency-rate prediction. "
    "Data source: BigQuery MLOps tables."
)


# =======================
# BigQuery Client & Cache
# =======================
@st.cache_data(ttl=300, show_spinner=False)
def get_bq_client():
    return bigquery.Client(project=PROJECT_ID)

def _to_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=True)
def load_training_runs() -> pd.DataFrame:
    """
    Expected schema (columns may be more than listed):
      - run_id (STRING)
      - model_name (STRING)
      - created_at (TIMESTAMP)
      - metrics (JSON string or STRUCT/JSON column) with keys in METRIC_KEYS
      - params (optional JSON)
    """
    client = get_bq_client()
    # Extract metrics from JSON; cast to FLOAT64
    # JSON_EXTRACT_SCALAR returns string; cast needed
    # Note: adjust column names if your schema differs
    metric_selects = ",\n".join(
        [
            f"CAST(JSON_EXTRACT_SCALAR(metrics, '$.{k}') AS FLOAT64) AS {k}"
            for k in METRIC_KEYS
        ]
    )

    sql = f"""
    SELECT
      run_id,
      model_name,
      {metric_selects},
      created_at
    FROM {TABLE_TRAINING_RUN}
    ORDER BY created_at DESC
    """
    try:
        df = client.query(sql).to_dataframe()
        # normalize model_name to lower snake (optional)
        if "model_name" in df.columns:
            df["model_name"] = df["model_name"].astype(str).str.strip().str.lower()
        # keep only supported models if you want
        return df
    except Exception as e:
        st.warning(f"Failed to load training runs: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=True)
def load_model_versions() -> pd.DataFrame:
    """
    Expected schema:
      - model_version_id
      - model_name
      - metrics_json (JSON with same keys)
      - status ("approved" / "candidate" / "deprecated")
      - created_at
    """
    client = get_bq_client()
    metric_selects = ",\n".join(
        [
            f"CAST(JSON_EXTRACT_SCALAR(metrics_json, '$.{k}') AS FLOAT64) AS {k}"
            for k in METRIC_KEYS
        ]
    )
    sql = f"""
    SELECT
      model_version_id,
      model_name,
      status,
      {metric_selects},
      created_at
    FROM {TABLE_MODEL_VERSION}
    ORDER BY created_at DESC
    """
    try:
        df = client.query(sql).to_dataframe()
        if "model_name" in df.columns:
            df["model_name"] = df["model_name"].astype(str).str.strip().str.lower()
        return df
    except Exception as e:
        st.warning(f"Failed to load model versions: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=True)
def load_deployments() -> pd.DataFrame:
    """
    Expected schema:
      - deployment_id
      - model_version_id
      - endpoint_url
      - traffic_split (FLOAT64)
      - deployed_at
    """
    client = get_bq_client()
    sql = f"""
    SELECT
      deployment_id,
      model_version_id,
      endpoint_url,
      traffic_split,
      deployed_at
    FROM {TABLE_DEPLOYMENT}
    ORDER BY deployed_at DESC
    """
    try:
        df = client.query(sql).to_dataframe()
        return df
    except Exception as e:
        st.warning(f"Failed to load deployments: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=True)
def load_predictions(selected_model: str) -> pd.DataFrame:
    """
    Expected table columns in PRED_TABLE:
      year, week, delinq, delinq_predicted, model_name (or model_version_id)
    """
    client = get_bq_client()

    # Try model_name first; if not present, drop filter
    base_sql = f"""
    SELECT
      year, week, delinq, delinq_predicted, 
      SAFE_CAST(model_name AS STRING) AS model_name
    FROM {PRED_TABLE}
    """
    # We'll try to filter by model_name if column exists; otherwise return all
    try:
        df = client.query(base_sql).to_dataframe()
        if "model_name" in df.columns:
            df["model_name"] = df["model_name"].astype(str).str.lower()
            if selected_model:
                df = df[df["model_name"] == selected_model.lower()]
        # add a date index if helpful (optional)
        return df
    except Exception as e:
        st.info(f"(Optional) Could not load predictions from {PRED_TABLE}: {e}")
        return pd.DataFrame()


# =======================
# Helpers / Computations
# =======================
def best_model_by(df_latest: pd.DataFrame) -> dict:
    """
    Choose best model based on lowest MAE primarily, break ties by highest R2.
    df_latest: one row per model with metric columns
    """
    if df_latest.empty:
        return {}
    temp = df_latest.copy()
    # replace None with +inf for MAE to avoid ranking issues
    temp["MAE_rank"] = temp["MAE"].rank(method="min", ascending=True, na_option="bottom")
    temp["R2_rank"] = temp["R2"].rank(method="min", ascending=False, na_option="bottom")
    temp["score"] = temp["MAE_rank"] + 0.5 * temp["R2_rank"]
    best = temp.sort_values(["score", "MAE", "R2"], ascending=[True, True, False]).head(1)
    return best.iloc[0].to_dict()

def latest_metrics_per_model(runs: pd.DataFrame) -> pd.DataFrame:
    """
    From runs (many rows per model), take the most recent row per model.
    """
    if runs.empty:
        return pd.DataFrame()
    runs = runs.sort_values("created_at", ascending=False)
    idx = runs.groupby("model_name", as_index=False)["created_at"].idxmax()
    latest = runs.loc[idx].copy()
    return latest

def deltas_against_previous(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delta metrics (latest vs previous run) per model for MAE and R2.
    Returns a df with columns model_name, MAE, MAE_delta, R2, R2_delta, created_at, etc.
    """
    if runs.empty:
        return pd.DataFrame()
    runs = runs.sort_values(["model_name", "created_at"])
    # compute deltas within each model
    def _add_deltas(g):
        g = g.sort_values("created_at")
        g["MAE_delta"] = g["MAE"].diff()
        g["R2_delta"] = g["R2"].diff()
        return g
    out = runs.groupby("model_name", group_keys=False).apply(_add_deltas)
    # keep only last row per model (latest)
    out_latest = out.groupby("model_name", as_index=False).tail(1)
    return out_latest


# ===========
# UI: Tabs
# ===========
tabs = st.tabs([
    "🏁 Overview / Model Registry",
    "📈 Model Performance Explorer",
    "📊 Metric Trends & Drift",
    "🚀 Deployment & Version History",
])

runs_df = load_training_runs()
versions_df = load_model_versions()
deploy_df = load_deployments()

# Normalize names for filtering consistency
if not runs_df.empty and "model_name" in runs_df.columns:
    runs_df["model_name"] = runs_df["model_name"].astype(str).str.strip().str.lower()
if not versions_df.empty and "model_name" in versions_df.columns:
    versions_df["model_name"] = versions_df["model_name"].astype(str).str.strip().str.lower()

# ------------------------------
# TAB 1: Overview / Model Registry
# ------------------------------
with tabs[0]:
    st.subheader("🏁 Overview / Model Registry")

    if runs_df.empty:
        st.warning("No training runs found in BigQuery. Once your Airflow pipeline registers runs, this page will populate.")
    else:
        latest = latest_metrics_per_model(runs_df)
        latest = latest[["model_name", "created_at"] + METRIC_KEYS].copy()

        # Compute deltas for KPI row
        deltas = deltas_against_previous(runs_df)[["model_name", "MAE_delta", "R2_delta"]]
        latest = latest.merge(deltas, on="model_name", how="left")

        # Determine best model
        best = best_model_by(latest) if not latest.empty else {}
        best_name = best.get("model_name", "—")
        best_mae = best.get("MAE", None)
        best_r2 = best.get("R2", None)

        # Join with versions to find currently approved version for each model
        approved_map = {}
        if not versions_df.empty:
            # prefer most recent approved version per model
            v = versions_df[versions_df["status"].str.lower() == "approved"].sort_values("created_at", ascending=False)
            approved_map = v.groupby("model_name").first()["model_version_id"].to_dict()

        # KPI row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏆 Best Model", best_name.upper() if isinstance(best_name, str) else "—")
        col2.metric("Best MAE ↓", f"{best_mae:.4f}" if best_mae is not None else "—")
        col3.metric("Best R² ↑", f"{best_r2:.3f}" if best_r2 is not None else "—")
        # show approved version for best model
        best_version = approved_map.get(best_name, "—")
        col4.metric("Approved Version (best)", best_version)

        # Comparison table
        table = latest.copy()
        table["Approved_Version"] = table["model_name"].map(approved_map).fillna("—")
        # prettier names
        table.rename(
            columns={
                "model_name": "Model",
                "created_at": "Last Trained",
                "Pearson_r": "Pearson r",
                "R2": "R²",
                "RMSE_recent6": "RMSE_recent6",
            },
            inplace=True,
        )
        # Order columns
        ordered_cols = [
            "Model", "Last Trained", "sMAPE", "RMSE_recent6", "MAE", "Pearson r", "R²", "MASE", "Approved_Version"
        ]
        table = table[ordered_cols]
        st.dataframe(table, use_container_width=True)

        # Bar chart comparison (MAE & RMSE_recent6)
        plot_df = latest.melt(
            id_vars=["model_name"],
            value_vars=["MAE", "RMSE_recent6"],
            var_name="metric", value_name="value"
        )
        fig = px.bar(
            plot_df,
            x="model_name", y="value", color="metric",
            barmode="group",
            title="MAE vs RMSE_recent6 by Model"
        )
        fig.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        # Trend (last 10 runs) – choose metric to display
        st.markdown("#### Metric Trend (last 10 runs)")
        metric_choice = st.selectbox(
            "Pick a metric", ["MAE", "RMSE_recent6", "R2", "sMAPE", "MASE", "Pearson_r"], index=0, key="overview_metric"
        )
        top10 = runs_df.sort_values("created_at", ascending=False).groupby("model_name").head(10)
        fig_trend = px.line(
            top10, x="created_at", y=metric_choice, color="model_name",
            title=f"{metric_choice} over recent runs"
        )
        fig_trend.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_trend, use_container_width=True)

# --------------------------------
# TAB 2: Model Performance Explorer
# --------------------------------
with tabs[1]:
    st.subheader("📈 Model Performance Explorer")
    if runs_df.empty:
        st.warning("No training runs found.")
    else:
        model_choice = st.selectbox("Select a model", MODEL_NAMES, index=1, key="explorer_model")
        mdf = runs_df[runs_df["model_name"] == model_choice]
        if mdf.empty:
            st.info(f"No runs found for model '{model_choice}'.")
        else:
            # Latest + last 3 runs
            mdf_sorted = mdf.sort_values("created_at", ascending=False).head(4)
            # Metrics grid
            latest_row = mdf_sorted.iloc[0]
            prev_row = mdf_sorted.iloc[1] if len(mdf_sorted) > 1 else None

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            def _delta(curr, prev, invert=False):
                if prev is None or curr is None or prev.get("value") is None:
                    return None
                d = curr["value"] - prev["value"]
                # For metrics where "lower is better" keep sign as-is; Streamlit will color appropriately
                # invert not used here; keeping simple deltas
                return round(d, 4)

            metrics_map = {
                "sMAPE": {"label": "sMAPE ↓", "value": _to_float(latest_row.get("sMAPE"))},
                "RMSE_recent6": {"label": "RMSE_recent6 ↓", "value": _to_float(latest_row.get("RMSE_recent6"))},
                "MAE": {"label": "MAE ↓", "value": _to_float(latest_row.get("MAE"))},
                "Pearson_r": {"label": "Pearson r ↑", "value": _to_float(latest_row.get("Pearson_r"))},
                "R2": {"label": "R² ↑", "value": _to_float(latest_row.get("R2"))},
                "MASE": {"label": "MASE ↓", "value": _to_float(latest_row.get("MASE"))},
            }

            prev_vals = {k: _to_float(prev_row.get(k)) if prev_row is not None else None for k in METRIC_KEYS}
            deltas = {k: (None if prev_vals[k] is None or metrics_map[k]["value"] is None else round(metrics_map[k]["value"] - prev_vals[k], 4))
                      for k in METRIC_KEYS}

            c1.metric(metrics_map["sMAPE"]["label"], f"{metrics_map['sMAPE']['value']}" if metrics_map["sMAPE"]["value"] is not None else "—", delta=deltas["sMAPE"])
            c2.metric(metrics_map["RMSE_recent6"]["label"], f"{metrics_map['RMSE_recent6']['value']}" if metrics_map["RMSE_recent6"]["value"] is not None else "—", delta=deltas["RMSE_recent6"])
            c3.metric(metrics_map["MAE"]["label"], f"{metrics_map['MAE']['value']}" if metrics_map["MAE"]["value"] is not None else "—", delta=deltas["MAE"])
            c4.metric(metrics_map["Pearson_r"]["label"], f"{metrics_map['Pearson_r']['value']}" if metrics_map["Pearson_r"]["value"] is not None else "—", delta=deltas["Pearson_r"])
            c5.metric(metrics_map["R2"]["label"], f"{metrics_map['R2']['value']}" if metrics_map["R2"]["value"] is not None else "—", delta=deltas["R2"])
            c6.metric(metrics_map["MASE"]["label"], f"{metrics_map['MASE']['value']}" if metrics_map["MASE"]["value"] is not None else "—", delta=deltas["MASE"])

            # Show recent runs table
            st.markdown("#### Recent Runs (latest 4)")
            show_cols = ["created_at"] + METRIC_KEYS
            st.dataframe(mdf_sorted[show_cols], use_container_width=True)

            # Actual vs Predicted (if available)
            st.markdown("#### Actual vs Predicted")
            preds = load_predictions(model_choice)
            if preds.empty:
                st.info("No prediction records found (optional). Populate the gold table to enable this chart.")
            else:
                # Create a sortable time index from year+week for plotting
                preds = preds.copy()
                if "year" in preds.columns and "week" in preds.columns:
                    preds["year_week"] = preds["year"].astype(str) + "-W" + preds["week"].astype(int).astype(str)
                fig_ap = px.line(preds.sort_values(["year", "week"]), x="year_week", y=["delinq", "delinq_predicted"],
                                 title=f"Actual vs Predicted — {model_choice}")
                fig_ap.update_layout(xaxis_title="", yaxis_title="Delinquency Rate")
                st.plotly_chart(fig_ap, use_container_width=True)

# -----------------------------------
# TAB 3: Metric Trends & Drift Analysis
# -----------------------------------
with tabs[2]:
    st.subheader("📊 Metric Trends & Drift")
    if runs_df.empty:
        st.warning("No training runs found.")
    else:
        metric_sel = st.selectbox("Pick a metric to visualize", ["MAE", "RMSE_recent6", "R2", "sMAPE", "MASE", "Pearson_r"], index=0)
        # Limit history window
        hist = runs_df.sort_values("created_at", ascending=True)
        # Optional filter: only show models that actually appear in data
        avail_models = sorted(hist["model_name"].dropna().unique().tolist())
        models_filter = st.multiselect("Models", options=avail_models, default=avail_models)
        hist = hist[hist["model_name"].isin(models_filter)]

        if hist.empty:
            st.info("No data after filtering.")
        else:
            fig_hist = px.line(hist, x="created_at", y=metric_sel, color="model_name",
                               title=f"{metric_sel} over time")
            fig_hist.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_hist, use_container_width=True)

        # Quick stability insight (variance of MAE per model)
        st.markdown("#### Stability Snapshot (lower variance in MAE = more stable)")
        stab = runs_df.groupby("model_name")["MAE"].agg(["mean", "std", "count"]).reset_index()
        stab.rename(columns={"mean": "MAE_mean", "std": "MAE_std", "count": "n_runs"}, inplace=True)
        st.dataframe(stab.sort_values("MAE_std"), use_container_width=True)

# ----------------------------------------
# TAB 4: Deployment & Version History
# ----------------------------------------
with tabs[3]:
    st.subheader("🚀 Deployment & Version History")
    if versions_df.empty and deploy_df.empty:
        st.warning("No version or deployment records found.")
    else:
        # Current deployed (traffic_split > 0)
        if not deploy_df.empty:
            live = deploy_df[deploy_df["traffic_split"] > 0].copy()
            if not live.empty:
                st.success("✅ Currently Deployed Models")
                st.dataframe(
                    live[["deployment_id", "model_version_id", "endpoint_url", "traffic_split", "deployed_at"]],
                    use_container_width=True
                )
            else:
                st.info("No active deployments with traffic_split > 0.")
        else:
            st.info("No deployment table data.")

        st.markdown("#### Model Versions")
        if not versions_df.empty:
            # Color status badges
            vdf = versions_df.copy()
            def _badge(s):
                s = (s or "").lower()
                if s == "approved":
                    return "🟢 approved"
                if s == "candidate":
                    return "🟠 candidate"
                if s == "deprecated":
                    return "🔴 deprecated"
                return s
            vdf["status_badge"] = vdf["status"].apply(_badge)
            show_cols = ["model_name", "model_version_id", "status_badge", "MAE", "R2", "created_at"]
            st.dataframe(vdf[show_cols].sort_values(["model_name", "created_at"], ascending=[True, False]),
                         use_container_width=True)

            # Simple timeline by created_at
            st.markdown("#### Version Timeline (by created_at)")
            fig_ver = px.scatter(
                vdf, x="created_at", y="model_name", color="status",
                hover_data=["model_version_id", "MAE", "R2"],
                title="Model versions over time"
            )
            st.plotly_chart(fig_ver, use_container_width=True)
        else:
            st.info("No model version records.")


# =================
# Metric Tooltips UI
# =================
with st.expander("ℹ️ Metric definitions & goals"):
    st.markdown("""
**Prediction Accuracy**
- **sMAPE** — symmetric mean absolute percentage error; lower is better.
- **RMSE_recent6** — RMSE over the last 6 weeks; lower is better (adaptiveness).

**Overall Fit and Correlation**
- **MAE** — mean absolute error; lower is better.
- **Pearson r** — correlation between actual and predicted; higher is better.

**Explanatory Power and Baseline Comparison**
- **R²** — proportion of variance explained; higher is better.
- **MASE** — error scaled by a baseline; lower is better.
    """)
