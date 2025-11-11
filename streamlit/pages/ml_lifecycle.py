# streamlit/pages/ml_lifecycle.py
# 🤖 ML Lifecycle & Model Registry — Credit Risk Intelligence Platform
# Prototype version: Displays all content and layout even when data is missing.

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
    st.caption("🚧 Prototype view — metrics will populate once MLOps pipeline writes to warehouse.")

    # ==================================
    # HELPER FUNCTIONS
    # ==================================
    def safe_display(x):
        """Return '-' if missing."""
        if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == "":
            return "–"
        return str(x)

    def safe_dataframe(df):
        """Return empty DataFrame with placeholder columns if missing."""
        if df is None or df.empty:
            return pd.DataFrame(columns=["model_name", *METRIC_KEYS, "created_at"])
        return df.fillna("–")

    @st.cache_resource(show_spinner=False)
    def get_bq_client():
        try:
            return bigquery.Client(project=PROJECT_ID)
        except Exception:
            return None

    # ==================================
    # TRY LOADING DATA (WILL FALLBACK TO PLACEHOLDER)
    # ==================================
    try:
        client = get_bq_client()
        if client:
            df_train = client.query(f"SELECT * FROM {TABLE_TRAINING_RUN} LIMIT 10").to_dataframe()
        else:
            df_train = pd.DataFrame()
    except Exception:
        df_train = pd.DataFrame()

    # If no data — build synthetic placeholder for display
    if df_train.empty:
        np.random.seed(42)
        df_train = pd.DataFrame({
            "model_name": MODEL_NAMES,
            "sMAPE": np.random.uniform(5, 15, len(MODEL_NAMES)),
            "RMSE_recent6": np.random.uniform(0.2, 1.0, len(MODEL_NAMES)),
            "MAE": np.random.uniform(0.1, 0.6, len(MODEL_NAMES)),
            "Pearson_r": np.random.uniform(0.7, 0.98, len(MODEL_NAMES)),
            "R2": np.random.uniform(0.65, 0.95, len(MODEL_NAMES)),
            "MASE": np.random.uniform(0.8, 1.2, len(MODEL_NAMES)),
            "created_at": [datetime(2025, 11, 1) + pd.Timedelta(days=i) for i in range(len(MODEL_NAMES))],
        })

    df_train = safe_dataframe(df_train)

    # ==================================
    # COMPUTE "BEST MODEL"
    # ==================================
    best_idx = df_train["MAE"].astype(float).idxmin()
    best_model = df_train.loc[best_idx]

    # ==================================
    # LAYOUT — 4 MAIN TABS
    # ==================================
    tabs = st.tabs([
        "🏁 Overview / Model Registry",
        "📈 Model Performance Explorer",
        "📊 Metric Trends & Drift",
        "🚀 Deployment & Version History",
    ])

    # --------------------------------------------------------
    # TAB 1 — OVERVIEW / MODEL REGISTRY
    # --------------------------------------------------------
    with tabs[0]:
        st.subheader("🏁 Overview / Model Registry")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏆 Best Model", safe_display(best_model["model_name"]))
        col2.metric("Best MAE ↓", round(float(best_model["MAE"]), 3))
        col3.metric("Best R² ↑", round(float(best_model["R2"]), 3))
        col4.metric("Status", "Approved")

        st.markdown("### 🔍 Latest Model Comparison")
        st.dataframe(df_train, use_container_width=True)

        # Comparison Chart
        st.markdown("#### 📊 Model Error Comparison")
        plot_df = df_train.melt(
            id_vars=["model_name"],
            value_vars=["MAE", "RMSE_recent6"],
            var_name="Metric",
            value_name="Value"
        )
        fig = px.bar(plot_df, x="model_name", y="Value", color="Metric", barmode="group",
                     title="Model Comparison by Error Metrics", text_auto=".2f")
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # TAB 2 — MODEL PERFORMANCE EXPLORER
    # --------------------------------------------------------
    with tabs[1]:
        st.subheader("📈 Model Performance Explorer")
        model_choice = st.selectbox("Select a model", MODEL_NAMES, index=0)
        model_data = df_train[df_train["model_name"] == model_choice]

        if model_data.empty:
            st.info("No recorded runs for this model (showing placeholder).")
            model_data = pd.DataFrame({
                "Metric": METRIC_KEYS,
                "Value": np.random.uniform(0.1, 1.0, len(METRIC_KEYS))
            })
            st.table(model_data)
        else:
            col = st.columns(6)
            for i, k in enumerate(METRIC_KEYS):
                val = safe_display(model_data.iloc[0][k])
                col[i % 6].metric(k, val)

            st.markdown("#### Recent Training History")
            st.dataframe(model_data, use_container_width=True)

            # Dummy Actual vs Predicted Plot
            t = np.arange(10)
            y_true = np.sin(t / 2) + np.random.normal(0, 0.1, len(t))
            y_pred = np.sin(t / 2 + 0.2)
            df_pred = pd.DataFrame({"Week": t, "Actual": y_true, "Predicted": y_pred})
            fig = px.line(df_pred, x="Week", y=["Actual", "Predicted"], title=f"Actual vs Predicted — {model_choice}")
            st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # TAB 3 — METRIC TRENDS & DRIFT
    # --------------------------------------------------------
    with tabs[2]:
        st.subheader("📊 Metric Trends & Drift Analysis")
        metric_sel = st.selectbox("Select Metric", METRIC_KEYS, index=0)
        df_m = df_train.copy()
        df_m = df_m.sort_values("created_at")
        fig = px.line(df_m, x="created_at", y=metric_sel, color="model_name",
                      title=f"{metric_sel} over time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        stability = df_train[["model_name", "MAE", "R2"]].copy()
        stability["Run Count"] = np.random.randint(3, 10, len(stability))
        st.markdown("#### Stability Summary")
        st.dataframe(stability, use_container_width=True)

    # --------------------------------------------------------
    # TAB 4 — DEPLOYMENT & VERSION HISTORY
    # --------------------------------------------------------
    with tabs[3]:
        st.subheader("🚀 Deployment & Version History")

        deploy_placeholder = pd.DataFrame({
            "deployment_id": [f"deploy_{m}" for m in MODEL_NAMES],
            "model_name": MODEL_NAMES,
            "endpoint_url": [f"https://api/{m}" for m in MODEL_NAMES],
            "traffic_split": np.round(np.random.uniform(0, 1, len(MODEL_NAMES)), 2),
            "deployed_at": pd.date_range("2025-10-01", periods=len(MODEL_NAMES), freq="3D")
        })

        st.markdown("### 🟢 Active Deployments")
        st.dataframe(deploy_placeholder, use_container_width=True)

        fig = px.scatter(deploy_placeholder, x="deployed_at", y="model_name",
                         size="traffic_split", color="traffic_split",
                         title="Deployment Timeline", hover_data=["endpoint_url"])
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # METRIC DEFINITIONS
    # --------------------------------------------------------
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
