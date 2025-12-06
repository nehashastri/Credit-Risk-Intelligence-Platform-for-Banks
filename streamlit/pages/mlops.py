# ============================================================
# STREAMLIT — ML LIFECYCLE & MODEL REGISTRY (REAL DATA VERSION)
# Wrapped inside nlopsnew() for integration with Home router.
# Supports your existing BigQuery tables exactly as they are.
# ============================================================

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from google.cloud import bigquery

# ------------------------------------------------------------
# MAIN FUNCTION (CALL THIS FROM HOME PAGE)
# ------------------------------------------------------------
st.write("🔧 nlopsnew() has started running")
def nlopsnew():

    # ------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------
    PROJECT_ID = os.getenv("GCP_PROJECT") or "pipeline-882-team-project"
    DATASET_MLOPS = "mlops"

    TABLE_TRAINING_RUN = f"`{PROJECT_ID}.{DATASET_MLOPS}.training_run`"
    TABLE_MODEL_VERSION = f"`{PROJECT_ID}.{DATASET_MLOPS}.model_version`"
    TABLE_DEPLOYMENT = f"`{PROJECT_ID}.{DATASET_MLOPS}.deployment`"

    METRIC_KEYS = ["MAE", "R2", "SMAPE", "RMSE_RECENT6", "PEARSON_R", "MASE"]

    # ------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------
    def parse_json(value):
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value)
        except:
            return {}

    def extract_metrics(df, col_name):
    # initialize all metric columns
        metric_map = {
            "mae": "MAE",
            "r2": "R2",
            "smape": "SMAPE",
            "rmse_recent6": "RMSE_RECENT6",
            "pearson_r": "PEARSON_R",
            "mase": "MASE"
        }

        # create empty columns
        for col in metric_map.values():
            df[col] = np.nan

        # fill values from JSON
        for idx, row in df.iterrows():
            data = parse_json(row.get(col_name))
            for key, val in data.items():
                key_lower = key.lower()
                if key_lower in metric_map:
                    df.at[idx, metric_map[key_lower]] = val

        return df


    def extract_model_name(df, col_name):
        df["model_name"] = None
        for idx, row in df.iterrows():
            params = parse_json(row.get(col_name))
            if isinstance(params, dict) and "algorithm" in params:
                df.at[idx, "model_name"] = params["algorithm"]
        return df

    def safe_display(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "–"
        return x

    @st.cache_resource(show_spinner=False)
    def get_bq_client():
        try:
            return bigquery.Client(project=PROJECT_ID)
        except:
            return None

    # ------------------------------------------------------------
    # PAGE HEADER
    # ------------------------------------------------------------
    st.title(" ML Lifecycle & Model Registry 🤖")
    st.caption(" Powered by real MLOps BigQuery tables — no placeholder data.")

    # ------------------------------------------------------------
    # LOAD BIGQUERY DATA
    # ------------------------------------------------------------
    client = get_bq_client()

    df_train = client.query(f"SELECT * FROM {TABLE_TRAINING_RUN}").to_dataframe()
    df_versions = client.query(f"SELECT * FROM {TABLE_MODEL_VERSION}").to_dataframe()
    df_deploy = client.query(f"SELECT * FROM {TABLE_DEPLOYMENT}").to_dataframe()

    # ------------------------------------------------------------
    # PARSE FIELDS
    # ------------------------------------------------------------
    df_train = extract_model_name(df_train, "params")
    df_train = extract_metrics(df_train, "metrics")

    df_versions = extract_model_name(df_versions, "params")
    df_versions = extract_metrics(df_versions, "metrics_json")

    # deployment → add model_name from model_version
    df_deploy = df_deploy.merge(
        df_versions[["model_version_id", "model_name"]],
        on="model_version_id",
        how="left"
    )

    # ------------------------------------------------------------
    # BEST MODEL CALCULATION
    # ------------------------------------------------------------
    df_train_clean = df_train.dropna(subset=["model_name"])
    df_train_clean = df_train_clean.sort_values("created_at", ascending=False)

    latest_runs = df_train_clean.groupby("model_name").first().reset_index()

    for m in METRIC_KEYS:
        latest_runs[m] = pd.to_numeric(latest_runs[m], errors="coerce")

    if not latest_runs.empty and latest_runs["MAE"].notna().any():
        best_model = latest_runs.loc[latest_runs["MAE"].idxmin()]
    else:
        best_model = None

    # ------------------------------------------------------------
    # TABS
    # ------------------------------------------------------------
    tabs = st.tabs([
        " Overview / Model Registry 🏁",
        " Model Performance Explorer 📈",
        " Metric Trends & Drift 📊",
        " Deployment & Version History 🚀"
    ])

    # ============================================================
    # TAB 1 — OVERVIEW
    # ============================================================
    with tabs[0]:
        st.subheader(" Overview / Model Registry 🏁")

        if best_model is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(" Best Model", best_model["model_name"])
            col2.metric("Best MAE ↓", safe_display(round(best_model["MAE"], 3)))
            col3.metric("Best R² ↑", safe_display(round(best_model["R2"], 3)))
            col4.metric("Status", "Best Recent Run")
        else:
            st.info("No training runs found.")

        st.markdown("### Latest Model Comparison 🔍")
        st.dataframe(latest_runs, use_container_width=True, hide_index=True)

        try:
            comparison = latest_runs[["model_name", "MAE", "RMSE_RECENT6"]].dropna()
            melt_df = comparison.melt(
                id_vars="model_name",
                value_vars=["MAE", "RMSE_RECENT6"],
                var_name="Metric",
                value_name="Value"
            )
            fig = px.bar(
                melt_df,
                x="model_name",
                y="Value",
                color="Metric",
                barmode="group",
                title="Model Comparison by Error Metrics"
            )
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Not enough metric data for comparison chart.")

    # ============================================================
    # TAB 2 — MODEL PERFORMANCE EXPLORER
    # ============================================================
    with tabs[1]:
        st.subheader(" Model Performance Explorer 📈")

        model_choices = sorted(df_train_clean["model_name"].unique())
        model_choice = st.selectbox("Select a model", model_choices)

        model_data = df_train_clean[df_train_clean["model_name"] == model_choice]

        if model_data.empty:
            st.info("No runs found for this model.")
        else:
            latest_model = model_data.sort_values("created_at", ascending=False).iloc[0]

            st.markdown("#### Latest Run Metrics")
            cols = st.columns(6)
            for i, m in enumerate(METRIC_KEYS):
                cols[i % 6].metric(m, safe_display(latest_model[m]))

            st.markdown("#### Recent Training History")
            st.dataframe(
                model_data.sort_values("created_at", ascending=False).head(10),
                use_container_width=True, hide_index=True
            )

    # ============================================================
    # TAB 3 — METRIC TRENDS & DRIFT
    # ============================================================
    with tabs[2]:
        st.subheader(" Metric Trends & Drift Analysis 📊")

        metric_sel = st.selectbox("Select Metric", METRIC_KEYS, index=0)

        df_m = df_train_clean.copy()
        df_m[metric_sel] = pd.to_numeric(df_m[metric_sel], errors="coerce")
        df_m = df_m.dropna(subset=[metric_sel])

        if df_m.empty:
            st.info("No data available for this metric.")
        else:
            fig = px.line(
                df_m,
                x="created_at",
                y=metric_sel,
                color="model_name",
                title=f"{metric_sel} Trend Over Time",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Model Stability Summary")
        stability = []
        for model in model_choices:
            sub = df_m[df_m["model_name"] == model][metric_sel]
            stability.append({
                "Model": model,
                "Count": len(sub),
                "Mean": round(sub.mean(), 4) if len(sub) else "–",
                "Std": round(sub.std(), 4) if len(sub) else "–"
            })
        st.dataframe(pd.DataFrame(stability), use_container_width=True, hide_index=True)

    # ============================================================
    # TAB 4 — DEPLOYMENT HISTORY
    # ============================================================
    with tabs[3]:
        st.subheader(" Deployment & Version History 🚀")

        if df_deploy.empty:
            st.info("No deployments found.")
        else:
            st.markdown("### Deployment Records 🟢")
            st.dataframe(df_deploy, use_container_width=True, hide_index=True)

            st.markdown("#### Deployment Timeline")
            try:
                fig = px.scatter(
                    df_deploy,
                    x="deployed_at",
                    y="model_name",
                    size="traffic_split",
                    hover_data=["endpoint_url", "deployment_id"]
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Unable to generate timeline plot.")
