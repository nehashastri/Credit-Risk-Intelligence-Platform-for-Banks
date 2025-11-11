# streamlit/pages/ml_lifecycle.py
# 🤖 ML Lifecycle & Model Registry — Credit Risk Intelligence Platform
# Prototype version: Displays all content and layout even when data is missing.

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

    def generate_placeholder_data():
        """Generate synthetic data for display when real data is unavailable."""
        np.random.seed(42)
        base_date = datetime(2025, 10, 1)
        
        # Training runs
        training_data = []
        for i, model in enumerate(MODEL_NAMES):
            for run in range(3):  # 3 runs per model
                training_data.append({
                    "run_id": f"run_{model}_{run}",
                    "model_name": model,
                    "sMAPE": round(np.random.uniform(5, 15), 2),
                    "RMSE_recent6": round(np.random.uniform(0.2, 1.0), 3),
                    "MAE": round(np.random.uniform(0.1, 0.6), 3),
                    "Pearson_r": round(np.random.uniform(0.7, 0.98), 3),
                    "R2": round(np.random.uniform(0.65, 0.95), 3),
                    "MASE": round(np.random.uniform(0.8, 1.2), 3),
                    "created_at": base_date + timedelta(days=i*7 + run*2),
                })
        
        df_train = pd.DataFrame(training_data)
        
        # Model versions
        version_data = []
        for i, model in enumerate(MODEL_NAMES):
            latest_run = df_train[df_train["model_name"] == model].iloc[-1]
            version_data.append({
                "model_version_id": f"v_{model}_1.0",
                "model_name": model,
                "status": np.random.choice(["approved", "pending", "rejected"], p=[0.6, 0.3, 0.1]),
                **{k: latest_run[k] for k in METRIC_KEYS},
                "created_at": latest_run["created_at"],
            })
        
        df_versions = pd.DataFrame(version_data)
        
        # Deployments
        deployment_data = []
        approved = df_versions[df_versions["status"] == "approved"]
        for idx, row in approved.iterrows():
            deployment_data.append({
                "deployment_id": f"deploy_{row['model_name']}",
                "model_version_id": row["model_version_id"],
                "model_name": row["model_name"],
                "endpoint_url": f"https://api.creditrisk.com/v1/predict/{row['model_name']}",
                "traffic_split": round(np.random.uniform(0.1, 1.0), 2),
                "deployed_at": row["created_at"] + timedelta(days=3),
            })
        
        df_deploy = pd.DataFrame(deployment_data)
        
        # Predictions
        prediction_data = []
        for week in range(1, 25):
            actual = 3.5 + 0.5 * np.sin(week / 4) + np.random.normal(0, 0.2)
            for model in MODEL_NAMES[:4]:  # Just a few models for predictions
                predicted = actual + np.random.normal(0, 0.3)
                prediction_data.append({
                    "year": 2025,
                    "week": week,
                    "model_name": model,
                    "delinq": round(actual, 3),
                    "delinq_predicted": round(predicted, 3),
                })
        
        df_preds = pd.DataFrame(prediction_data)
        
        return df_train, df_versions, df_deploy, df_preds

    # ==================================
    # TRY LOADING DATA (WILL FALLBACK TO PLACEHOLDER)
    # ==================================
    use_placeholder = False
    
    try:
        client = get_bq_client()
        if client:
            # Try to load training runs
            df_train = client.query(f"SELECT * FROM {TABLE_TRAINING_RUN} LIMIT 100").to_dataframe()
            if df_train.empty:
                use_placeholder = True
        else:
            use_placeholder = True
    except Exception as e:
        st.info(f"📊 Using placeholder data (BigQuery connection issue: {str(e)[:100]})")
        use_placeholder = True

    if use_placeholder:
        df_train, df_versions, df_deploy, df_preds = generate_placeholder_data()
    else:
        # Load real data
        try:
            df_versions = client.query(f"SELECT * FROM {TABLE_MODEL_VERSION}").to_dataframe()
            df_deploy = client.query(f"SELECT * FROM {TABLE_DEPLOYMENT}").to_dataframe()
            df_preds = client.query(f"SELECT * FROM {PRED_TABLE} LIMIT 1000").to_dataframe()
        except Exception:
            df_train, df_versions, df_deploy, df_preds = generate_placeholder_data()
            use_placeholder = True

    # Ensure data is clean
    df_train = safe_dataframe(df_train)
    df_versions = safe_dataframe(df_versions) if not df_versions.empty else pd.DataFrame()
    df_deploy = safe_dataframe(df_deploy) if not df_deploy.empty else pd.DataFrame()
    df_preds = safe_dataframe(df_preds) if not df_preds.empty else pd.DataFrame()

    # ==================================
    # COMPUTE "BEST MODEL"
    # ==================================
    latest_runs = df_train.sort_values("created_at", ascending=False).groupby("model_name").first().reset_index()
    
    # Convert metrics to numeric for comparison
    for metric in METRIC_KEYS:
        latest_runs[metric] = pd.to_numeric(latest_runs[metric], errors='coerce')
    
    # Find best model (lowest MAE)
    valid_mae = latest_runs[latest_runs["MAE"].notna()]
    if not valid_mae.empty:
        best_idx = valid_mae["MAE"].idxmin()
        best_model = valid_mae.loc[best_idx]
    else:
        best_model = latest_runs.iloc[0]

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
        
        if use_placeholder:
            st.info("💡 Showing simulated data. Once MLOps pipeline runs, real metrics will appear here.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏆 Best Model", safe_display(best_model["model_name"]))
        col2.metric("Best MAE ↓", safe_display(round(float(best_model["MAE"]), 3) if pd.notna(best_model["MAE"]) else "–"))
        col3.metric("Best R² ↑", safe_display(round(float(best_model["R2"]), 3) if pd.notna(best_model["R2"]) else "–"))
        col4.metric("Status", "✅ Approved" if not use_placeholder else "🟡 Pending")

        st.markdown("### 🔍 Latest Model Comparison")
        
        # Display table
        display_cols = ["model_name"] + METRIC_KEYS + ["created_at"]
        available_cols = [col for col in display_cols if col in latest_runs.columns]
        st.dataframe(latest_runs[available_cols], use_container_width=True, hide_index=True)

        # Comparison Chart
        st.markdown("#### 📊 Model Error Comparison")
        try:
            plot_data = latest_runs[["model_name", "MAE", "RMSE_recent6"]].copy()
            plot_data["MAE"] = pd.to_numeric(plot_data["MAE"], errors='coerce')
            plot_data["RMSE_recent6"] = pd.to_numeric(plot_data["RMSE_recent6"], errors='coerce')
            plot_data = plot_data.dropna()
            
            if not plot_data.empty:
                plot_df = plot_data.melt(
                    id_vars=["model_name"],
                    value_vars=["MAE", "RMSE_recent6"],
                    var_name="Metric",
                    value_name="Value"
                )
                fig = px.bar(
                    plot_df, 
                    x="model_name", 
                    y="Value", 
                    color="Metric", 
                    barmode="group",
                    title="Model Comparison by Error Metrics",
                    text_auto=".3f",
                    color_discrete_map={"MAE": "#FF6B6B", "RMSE_recent6": "#4ECDC4"}
                )
                fig.update_layout(xaxis_title="Model", yaxis_title="Error Value")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for chart visualization.")
        except Exception as e:
            st.warning(f"Chart unavailable: {str(e)}")

    # --------------------------------------------------------
    # TAB 2 — MODEL PERFORMANCE EXPLORER
    # --------------------------------------------------------
    with tabs[1]:
        st.subheader("📈 Model Performance Explorer")
        model_choice = st.selectbox("Select a model", MODEL_NAMES, index=0)
        
        model_data = df_train[df_train["model_name"] == model_choice].sort_values("created_at", ascending=False)

        if model_data.empty:
            st.info(f"No recorded runs for '{model_choice}' yet. Data will appear after pipeline execution.")
        else:
            # Latest metrics
            latest_model = model_data.iloc[0]
            
            st.markdown("#### Latest Run Metrics")
            col = st.columns(6)
            for i, k in enumerate(METRIC_KEYS):
                val = safe_display(latest_model.get(k, "–"))
                col[i % 6].metric(k, val)

            st.markdown("#### Recent Training History")
            history_display = model_data.head(10)
            st.dataframe(history_display, use_container_width=True, hide_index=True)

            # Actual vs Predicted Plot
            st.markdown("#### 📉 Actual vs Predicted Delinquency Rate")
            model_preds = df_preds[df_preds["model_name"] == model_choice]
            
            if not model_preds.empty:
                try:
                    model_preds = model_preds.copy()
                    model_preds["delinq"] = pd.to_numeric(model_preds["delinq"], errors='coerce')
                    model_preds["delinq_predicted"] = pd.to_numeric(model_preds["delinq_predicted"], errors='coerce')
                    model_preds = model_preds.dropna(subset=["delinq", "delinq_predicted"])
                    model_preds = model_preds.sort_values(["year", "week"])
                    model_preds["time_label"] = model_preds["year"].astype(str) + "-W" + model_preds["week"].astype(str).str.zfill(2)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=model_preds["time_label"],
                        y=model_preds["delinq"],
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color='#2E86AB', width=2),
                        marker=dict(size=6)
                    ))
                    fig.add_trace(go.Scatter(
                        x=model_preds["time_label"],
                        y=model_preds["delinq_predicted"],
                        mode='lines+markers',
                        name='Predicted',
                        line=dict(color='#F24236', width=2, dash='dash'),
                        marker=dict(size=6)
                    ))
                    fig.update_layout(
                        title=f"Actual vs Predicted — {model_choice}",
                        xaxis_title="Week",
                        yaxis_title="Delinquency Rate (%)",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate prediction plot: {str(e)}")
            else:
                st.info("Prediction data not yet available for this model.")

    # --------------------------------------------------------
    # TAB 3 — METRIC TRENDS & DRIFT
    # --------------------------------------------------------
    with tabs[2]:
        st.subheader("📊 Metric Trends & Drift Analysis")
        
        metric_sel = st.selectbox("Select Metric", METRIC_KEYS, index=0)
        
        # Prepare data
        df_m = df_train.copy()
        df_m[metric_sel] = pd.to_numeric(df_m[metric_sel], errors='coerce')
        df_m = df_m.dropna(subset=[metric_sel])
        df_m = df_m.sort_values("created_at")
        
        if not df_m.empty:
            fig = px.line(
                df_m, 
                x="created_at", 
                y=metric_sel, 
                color="model_name",
                title=f"{metric_sel} Trend Over Time",
                markers=True,
                line_shape="spline"
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title=metric_sel,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No data available for {metric_sel} yet.")

        st.markdown("#### Model Stability Summary")
        
        # Calculate stability metrics
        stability_data = []
        for model in MODEL_NAMES:
            model_runs = df_train[df_train["model_name"] == model]
            if not model_runs.empty:
                mae_vals = pd.to_numeric(model_runs["MAE"], errors='coerce').dropna()
                r2_vals = pd.to_numeric(model_runs["R2"], errors='coerce').dropna()
                
                stability_data.append({
                    "Model": model,
                    "Runs": len(model_runs),
                    "MAE Mean": round(mae_vals.mean(), 3) if not mae_vals.empty else "–",
                    "MAE Std": round(mae_vals.std(), 3) if not mae_vals.empty else "–",
                    "R² Mean": round(r2_vals.mean(), 3) if not r2_vals.empty else "–",
                    "R² Std": round(r2_vals.std(), 3) if not r2_vals.empty else "–",
                })
        
        if stability_data:
            stability_df = pd.DataFrame(stability_data)
            st.dataframe(stability_df, use_container_width=True, hide_index=True)
        else:
            st.info("Stability metrics will appear after multiple training runs.")

    # --------------------------------------------------------
    # TAB 4 — DEPLOYMENT & VERSION HISTORY
    # --------------------------------------------------------
    with tabs[3]:
        st.subheader("🚀 Deployment & Version History")

        # Active Deployments
        if not df_deploy.empty:
            st.markdown("### 🟢 Active Deployments")
            
            # Filter for active deployments
            df_deploy["traffic_split"] = pd.to_numeric(df_deploy["traffic_split"], errors='coerce')
            active_deploys = df_deploy[df_deploy["traffic_split"] > 0].sort_values("deployed_at", ascending=False)
            
            if not active_deploys.empty:
                st.dataframe(active_deploys, use_container_width=True, hide_index=True)
                
                # Deployment timeline
                st.markdown("#### Deployment Timeline")
                try:
                    fig = px.scatter(
                        active_deploys,
                        x="deployed_at",
                        y="model_name",
                        size="traffic_split",
                        color="traffic_split",
                        hover_data=["endpoint_url", "deployment_id"],
                        title="Model Deployment Timeline",
                        color_continuous_scale="Viridis"
                    )
                    fig.update_layout(
                        xaxis_title="Deployment Date",
                        yaxis_title="Model",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Timeline visualization unavailable: {str(e)}")
            else:
                st.info("No active deployments found.")
        else:
            st.info("Deployment data will populate after models are deployed.")

        # Model Version History
        if not df_versions.empty:
            st.markdown("### 📋 Model Version History")
            
            version_display = df_versions[["model_name", "status", "MAE", "R2", "created_at"]].copy()
            st.dataframe(version_display, use_container_width=True, hide_index=True)
            
            # Version status distribution
            st.markdown("#### Version Status Distribution")
            status_counts = df_versions["status"].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Model Version Status",
                color_discrete_map={"approved": "#28a745", "pending": "#ffc107", "rejected": "#dc3545"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Version history will appear after model versioning is implemented.")

    # --------------------------------------------------------
    # METRIC DEFINITIONS
    # --------------------------------------------------------
    with st.expander("ℹ️ Metric Definitions"):
        st.markdown("""
**Prediction Accuracy**
- **sMAPE** — Symmetric Mean Absolute Percentage Error; lower is better. Measures percentage error symmetrically.
- **RMSE_recent6** — Root Mean Squared Error over recent 6 weeks; lower is better. Emphasizes larger errors.

**Overall Fit and Correlation**
- **MAE** — Mean Absolute Error; lower is better. Average magnitude of prediction errors.
- **Pearson r** — Correlation between actual and predicted; higher is better (range: -1 to 1).

**Explanatory Power & Baseline Comparison**
- **R²** — Coefficient of determination; higher is better (range: 0 to 1). Proportion of variance explained.
- **MASE** — Mean Absolute Scaled Error; lower is better. Compares performance to naive baseline.
""")

# Call the function when the page is loaded
if __name__ == "__main__":
    show_ml_lifecycle()
