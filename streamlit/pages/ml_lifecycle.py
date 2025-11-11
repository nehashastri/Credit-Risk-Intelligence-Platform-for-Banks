# streamlit/pages/ml_lifecycle.py
# 🤖 ML Lifecycle & Model Registry  — Credit Risk Intelligence Platform
# Tabs:
#   1) Overview / Model Registry
#   2) Model Performance Explorer
#   3) Metric Trends & Drift Analysis
#   4) Deployment & Version History

import os
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from google.cloud import bigquery

# ==================================
# CONFIGURATION
# ==================================
PROJECT_ID = os.getenv("GCP_PROJECT") or "pipeline-882-team-project"
DATASET_MLOPS = "mlops"
TABLE_TRAINING_RUN = f"`{PROJECT_ID}.{DATASET_MLOPS}.training_run`"
TABLE_MODEL_VERSION = f"`{PROJECT_ID}.{DATASET_MLOPS}.model_version`"
TABLE_DEPLOYMENT = f"`{PROJECT_ID}.{DATASET_MLOPS}.deployment`"
PRED_TABLE = f"`{PROJECT_ID}.gold.fact_delinquency_predictions`"

MODEL_NAMES = [
    "base", "linear_regression", "elastic_net",
    "random_forest", "gradient_boosting", "xgboost", "lightgbm",
]

METRIC_KEYS = ["sMAPE", "RMSE_recent6", "MAE", "Pearson_r", "R2", "MASE"]

# ==================================
# PAGE CONFIG
# ==================================
def show_ml_lifecycle():
    st.title("🤖 ML Lifecycle & Model Registry")
    st.info("🚧 Prototype view — metrics will populate once MLOps pipeline writes to warehouse.")

    # ==================================
    # HELPERS
    # ==================================
    def safe_float(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return None
            return float(x)
        except Exception:
            return None
    
    def safe_display(x):
        """Return '-' if missing, else string value."""
        if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == "":
            return "–"
        return str(x)
    
    def safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Replace NaN/None with '-' for display."""
        if df is None or df.empty:
            return pd.DataFrame()
        return df.fillna("–")
    
    @st.cache_resource(show_spinner=False)
    def get_bq_client():
        try:
            return bigquery.Client(project=PROJECT_ID)
        except Exception as e:
            st.warning(f"⚠️ Could not connect to BigQuery: {e}")
            return None
    
    # ==================================
    # DATA LOADERS WITH TRY/EXCEPT
    # ==================================
    @st.cache_data(ttl=300, show_spinner=True)
    def load_training_runs():
        client = get_bq_client()
        if client is None:
            return pd.DataFrame()
        metric_selects = ",\n".join(
            [f"SAFE_CAST(JSON_EXTRACT_SCALAR(metrics, '$.{k}') AS FLOAT64) AS {k}" for k in METRIC_KEYS]
        )
        sql = f"""
        SELECT run_id, SAFE_CAST(model_name AS STRING) AS model_name,
               {metric_selects}, SAFE_CAST(created_at AS TIMESTAMP) AS created_at
        FROM {TABLE_TRAINING_RUN}
        ORDER BY created_at DESC
        """
        try:
            df = client.query(sql).to_dataframe()
            if not df.empty:
                df["model_name"] = df["model_name"].astype(str).str.lower().str.strip()
            return df
        except Exception as e:
            st.warning(f"⚠️ Could not load training_run table: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=300, show_spinner=True)
    def load_model_versions():
        client = get_bq_client()
        if client is None:
            return pd.DataFrame()
        metric_selects = ",\n".join(
            [f"SAFE_CAST(JSON_EXTRACT_SCALAR(metrics_json, '$.{k}') AS FLOAT64) AS {k}" for k in METRIC_KEYS]
        )
        sql = f"""
        SELECT model_version_id, SAFE_CAST(model_name AS STRING) AS model_name,
               SAFE_CAST(status AS STRING) AS status,
               {metric_selects}, SAFE_CAST(created_at AS TIMESTAMP) AS created_at
        FROM {TABLE_MODEL_VERSION}
        ORDER BY created_at DESC
        """
        try:
            df = client.query(sql).to_dataframe()
            if not df.empty:
                df["model_name"] = df["model_name"].astype(str).str.lower().str.strip()
            return df
        except Exception as e:
            st.warning(f"⚠️ Could not load model_version table: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=300, show_spinner=True)
    def load_deployments():
        client = get_bq_client()
        if client is None:
            return pd.DataFrame()
        sql = f"""
        SELECT deployment_id, model_version_id, endpoint_url,
               SAFE_CAST(traffic_split AS FLOAT64) AS traffic_split,
               SAFE_CAST(deployed_at AS TIMESTAMP) AS deployed_at
        FROM {TABLE_DEPLOYMENT}
        ORDER BY deployed_at DESC
        """
        try:
            return client.query(sql).to_dataframe()
        except Exception as e:
            st.warning(f"⚠️ Could not load deployment table: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=300, show_spinner=False)
    def load_predictions(model_name: str):
        client = get_bq_client()
        if client is None:
            return pd.DataFrame()
        sql = f"""
        SELECT SAFE_CAST(year AS INT64) AS year,
               SAFE_CAST(week AS INT64) AS week,
               SAFE_CAST(delinq AS FLOAT64) AS delinq,
               SAFE_CAST(delinq_predicted AS FLOAT64) AS delinq_predicted,
               SAFE_CAST(model_name AS STRING) AS model_name
        FROM {PRED_TABLE}
        """
        try:
            df = client.query(sql).to_dataframe()
            if not df.empty:
                df["model_name"] = df["model_name"].astype(str).str.lower()
                df = df[df["model_name"] == model_name.lower()]
            return df
        except Exception as e:
            st.info(f"(Optional) Could not load prediction table: {e}")
            return pd.DataFrame()
    
    # ==================================
    # COMPUTATION HELPERS
    # ==================================
    def latest_metrics(df):
        if df.empty:
            return pd.DataFrame()
        return df.sort_values("created_at", ascending=False).groupby("model_name").head(1)
    
    def best_model(df):
        if df.empty:
            return {}
        df2 = df.copy()
        df2["MAE_rank"] = df2["MAE"].rank(ascending=True, method="min")
        df2["R2_rank"] = df2["R2"].rank(ascending=False, method="min")
        df2["score"] = df2["MAE_rank"] + 0.5 * df2["R2_rank"]
        return df2.sort_values("score").iloc[0].to_dict()
    
    # ==================================
    # LOAD DATA
    # ==================================
    runs_df = load_training_runs()
    versions_df = load_model_versions()
    deploy_df = load_deployments()
    
    # ==================================
    # UI LAYOUT — 4 TABS
    # ==================================
    tabs = st.tabs([
        "🏁 Overview / Model Registry",
        "📈 Model Performance Explorer",
        "📊 Metric Trends & Drift",
        "🚀 Deployment & Version History",
    ])
    
    # --------------------------------------------------------
    # TAB 1 — OVERVIEW / REGISTRY
    # --------------------------------------------------------
    with tabs[0]:
        st.subheader("🏁 Overview / Model Registry")
        if runs_df.empty:
            st.warning("No data available in training_run table.")
        else:
            latest = latest_metrics(runs_df)
            best = best_model(latest)
            best_name = safe_display(best.get("model_name"))
            best_mae = safe_display(best.get("MAE"))
            best_r2 = safe_display(best.get("R2"))
    
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🏆 Best Model", best_name)
            c2.metric("Best MAE ↓", best_mae)
            c3.metric("Best R² ↑", best_r2)
            c4.metric("Approved Version", safe_display("—"))
    
            df_show = latest.copy()
            df_show = safe_dataframe(df_show)
            st.dataframe(df_show, use_container_width=True)
    
            # Chart (skip if data missing)
            if not df_show.empty:
                try:
                    plot_df = df_show.melt(
                        id_vars=["model_name"], value_vars=["MAE", "RMSE_recent6"],
                        var_name="metric", value_name="value"
                    )
                    plot_df = plot_df.dropna(subset=["value"])
                    fig = px.bar(plot_df, x="model_name", y="value", color="metric", barmode="group")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Chart unavailable due to incomplete data.")
    
    # --------------------------------------------------------
    # TAB 2 — MODEL PERFORMANCE EXPLORER
    # --------------------------------------------------------
    with tabs[1]:
        st.subheader("📈 Model Performance Explorer")
        if runs_df.empty:
            st.warning("No model runs available.")
        else:
            model_choice = st.selectbox("Select a model", MODEL_NAMES)
            df_m = runs_df[runs_df["model_name"] == model_choice]
            if df_m.empty:
                st.info("No data found for this model.")
            else:
                latest_row = df_m.sort_values("created_at", ascending=False).head(1).iloc[0]
                col = st.columns(6)
                for i, k in enumerate(METRIC_KEYS):
                    val = safe_display(latest_row.get(k))
                    col[i % 6].metric(f"{k}", val)
    
                st.dataframe(safe_dataframe(df_m.head(5)), use_container_width=True)
    
                preds = load_predictions(model_choice)
                if preds.empty:
                    st.info("Prediction data not available.")
                else:
                    try:
                        preds["t"] = preds["year"].astype(str) + "-W" + preds["week"].astype(str)
                        fig = px.line(preds, x="t", y=["delinq", "delinq_predicted"], title=f"Actual vs Predicted — {model_choice}")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.info("Could not plot Actual vs Predicted due to incomplete data.")
    
    # --------------------------------------------------------
    # TAB 3 — METRIC TRENDS & DRIFT
    # --------------------------------------------------------
    with tabs[2]:
        st.subheader("📊 Metric Trends & Drift Analysis")
        if runs_df.empty:
            st.warning("No training history available.")
        else:
            metric_sel = st.selectbox("Select Metric", METRIC_KEYS, index=0)
            df_m = runs_df.dropna(subset=[metric_sel])
            if df_m.empty:
                st.info("No values available for this metric.")
            else:
                fig = px.line(df_m, x="created_at", y=metric_sel, color="model_name", title=f"{metric_sel} over time")
                st.plotly_chart(fig, use_container_width=True)
    
            # Stability table
            stab = (
                runs_df.groupby("model_name")[["MAE"]]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            stab.columns = ["Model", "MAE_mean", "MAE_std", "n_runs"]
            st.dataframe(safe_dataframe(stab), use_container_width=True)
    
    # --------------------------------------------------------
    # TAB 4 — DEPLOYMENT & VERSION HISTORY
    # --------------------------------------------------------
    with tabs[3]:
        st.subheader("🚀 Deployment & Version History")
        if versions_df.empty and deploy_df.empty:
            st.warning("No deployment or version records found.")
        else:
            if not deploy_df.empty:
                live = deploy_df[deploy_df.get("traffic_split", 0) > 0]
                if not live.empty:
                    st.success("✅ Active Deployments")
                    st.dataframe(safe_dataframe(live), use_container_width=True)
                else:
                    st.info("No currently active deployments.")
            else:
                st.info("Deployment table missing or empty.")
    
            if not versions_df.empty:
                vdf = safe_dataframe(versions_df)
                st.dataframe(vdf[["model_name", "status", "MAE", "R2", "created_at"]], use_container_width=True)
                try:
                    fig = px.scatter(vdf, x="created_at", y="model_name", color="status",
                                     hover_data=["model_version_id"], title="Model Versions over Time")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Could not render version timeline.")
            else:
                st.info("Model version table missing or empty.")
    
    # ==================================
    # METRIC DEFINITIONS
    # ==================================
    with st.expander("ℹ️ Metric definitions"):
        st.markdown("""
    **Prediction Accuracy**
    - sMAPE — symmetric mean absolute percentage error; lower is better.
    - RMSE_recent6 — RMSE over recent 6 weeks; lower is better.
    
    **Overall Fit and Correlation**
    - MAE — mean absolute error; lower is better.
    - Pearson r — correlation between actual and predicted; higher is better.
    
    **Explanatory Power & Baseline Comparison**
    - R² — variance explained; higher is better.
    - MASE — mean absolute scaled error; lower is better.
        """)
