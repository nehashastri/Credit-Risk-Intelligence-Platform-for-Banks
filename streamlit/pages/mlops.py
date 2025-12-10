# ============================================================
# STREAMLIT — ML LIFECYCLE & MODEL REGISTRY (REAL DATA VERSION)
# ============================================================

import os
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from utils.mlops_utils import (
    parse_json,
    safe_display,
    get_bq_client,
    extract_metrics,
    extract_model_name,
    format_datetime_column,
    get_all_metrics_from_row,
    METRIC_KEYS,
    METRIC_KEY_MAPPING
)

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

    # METRIC_KEYS imported from utils.mlops_utils

    # ------------------------------------------------------------
    # PAGE HEADER
    # ------------------------------------------------------------
    st.title(" ML Lifecycle & Model Registry 🤖")
    st.caption(" Powered by MLOps BigQuery tables.")

    # ------------------------------------------------------------
    # LOAD BIGQUERY DATA
    # ------------------------------------------------------------
    client = get_bq_client(PROJECT_ID)

    df_train = client.query(f"SELECT * FROM {TABLE_TRAINING_RUN}").to_dataframe()
    df_versions = client.query(f"SELECT * FROM {TABLE_MODEL_VERSION}").to_dataframe()
    df_deploy = client.query(f"SELECT * FROM {TABLE_DEPLOYMENT}").to_dataframe()

    # ------------------------------------------------------------
    # PARSE FIELDS
    # ------------------------------------------------------------
    df_train = extract_model_name(df_train, "params")
    df_train = extract_metrics(df_train, "metrics")
    
    # Parse created_at column to datetime (keep as datetime for sorting)
    if "created_at" in df_train.columns:
        df_train["created_at"] = pd.to_datetime(df_train["created_at"], errors="coerce")

    df_versions = extract_model_name(df_versions, "params")
    df_versions = extract_metrics(df_versions, "metrics_json")

    # Get model_name from training_run via model_version
    # Link through run_id if available in model_version
    if "run_id" in df_versions.columns and "run_id" in df_train.columns:
        # Merge model_version with training_run to get model_name
        df_versions = df_versions.merge(
            df_train[["run_id", "model_name"]],
            on="run_id",
            how="left",
            suffixes=("_from_params", "_from_train")
        )
        # Use model_name from training_run if available, otherwise use from model_version params
        if "model_name_from_train" in df_versions.columns:
            # Fill model_name_from_train with model_name_from_params where missing
            df_versions["model_name"] = df_versions["model_name_from_train"].fillna(
                df_versions["model_name_from_params"]
            )
            # Drop the intermediate columns
            df_versions = df_versions.drop(columns=["model_name_from_params", "model_name_from_train"], errors="ignore")
        elif "model_name_from_params" in df_versions.columns:
            # If only model_name_from_params exists, use it
            df_versions["model_name"] = df_versions["model_name_from_params"]
            df_versions = df_versions.drop(columns=["model_name_from_params"], errors="ignore")
    
    # deployment → add model_name from model_version
    df_deploy = df_deploy.merge(
        df_versions[["model_version_id", "model_name"]],
        on="model_version_id",
        how="left"
    )
    
    # Parse deployed_at column to datetime (keep as datetime for sorting)
    if "deployed_at" in df_deploy.columns:
        df_deploy["deployed_at"] = pd.to_datetime(df_deploy["deployed_at"], errors="coerce")
    
    # Parse created_at in df_versions for display
    if "created_at" in df_versions.columns:
        df_versions["created_at"] = pd.to_datetime(df_versions["created_at"], errors="coerce")
    
    # Extract metrics from deployment table
    df_deploy = extract_metrics(df_deploy, "metrics")

    # ------------------------------------------------------------
    # BEST MODEL CALCULATION - FROM DEPLOYMENT TABLE
    # ------------------------------------------------------------
    # Find the currently deployed model (traffic_split > 0.0)
    if "traffic_split" in df_deploy.columns:
        df_deploy["traffic_split"] = pd.to_numeric(df_deploy["traffic_split"], errors="coerce")
        active_deployments = df_deploy[df_deploy["traffic_split"] > 0.0].copy()
        
        if not active_deployments.empty:
            # Get the most recently deployed model with traffic_split > 0
            active_deployments = active_deployments.sort_values("deployed_at", ascending=False)
            best_model_row = active_deployments.iloc[0]
            
            # Convert metrics to numeric in the dataframe first
            for m in METRIC_KEYS:
                if m in active_deployments.columns:
                    active_deployments[m] = pd.to_numeric(active_deployments[m], errors="coerce")
            
            # Get the best model row again after numeric conversion
            best_model = active_deployments.iloc[0]
        else:
            best_model = None
    else:
        best_model = None
    
    # Keep latest_runs calculation for the comparison table
    df_train_clean = df_train.dropna(subset=["model_name"])
    df_train_clean = df_train_clean.sort_values("created_at", ascending=False)

    latest_runs = df_train_clean.groupby("model_name").first().reset_index()

    for m in METRIC_KEYS:
        latest_runs[m] = pd.to_numeric(latest_runs[m], errors="coerce")

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
            # Get model_name - best_model is a pandas Series from iloc[0]
            model_name = None
            if "model_name" in best_model.index:
                model_name = best_model["model_name"]
            
            # If model_name is still None or empty, try to get from model_version_id
            if not model_name or (isinstance(model_name, str) and model_name.lower() == "none") or pd.isna(model_name):
                if "model_version_id" in best_model.index:
                    model_version_id = best_model["model_version_id"]
                    if model_version_id:
                        # Try to get from df_versions
                        version_match = df_versions[df_versions["model_version_id"] == model_version_id]
                        if not version_match.empty and "model_name" in version_match.columns:
                            model_name = version_match.iloc[0]["model_name"]
            
            # Final fallback
            if not model_name or (isinstance(model_name, str) and model_name.lower() == "none") or pd.isna(model_name):
                model_name = "Unknown Model"
            
            st.markdown(f"#### 🏆 Best Model: **{model_name}**")
            
            # Get all available metrics from the best_model using utils function
            available_metrics = get_all_metrics_from_row(best_model, "metrics", METRIC_KEY_MAPPING)
            
            # Display metrics in columns (3 metrics per row)
            if available_metrics:
                # Filter out non-numeric metrics like test_count
                numeric_metrics = {}
                for key, val in available_metrics.items():
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        # Skip test_count as it's not a performance metric
                        if key.lower() != "test_count":
                            numeric_metrics[key] = val
                
                if numeric_metrics:
                    num_cols = 3
                    metric_items = list(numeric_metrics.items())
                    
                    for i in range(0, len(metric_items), num_cols):
                        cols = st.columns(num_cols)
                        for j, (metric_name, metric_value) in enumerate(metric_items[i:i+num_cols]):
                            if j < num_cols:
                                with cols[j]:
                                    # Format metric name for display
                                    display_name = metric_name.replace("_", " ").title()
                                    # Add arrows for better/worse indicators
                                    metric_lower = metric_name.lower()
                                    if metric_lower in ["mae", "smape", "rmse_recent6", "mase", "rmse"]:
                                        display_name += " ↓"
                                    elif metric_lower in ["r2", "pearson_r", "r²"]:
                                        display_name += " ↑"
                                    
                                    st.metric(display_name, safe_display(round(metric_value, 4) if isinstance(metric_value, (int, float)) else metric_value))
                else:
                    st.info("No numeric metrics available for the deployed model.")
            else:
                st.info("No metrics available for the deployed model.")
        else:
            st.info("No active deployment found (traffic_split > 0).")

        st.markdown("### Latest Model Comparison 🔍")
        
        # Format created_at for display and select only the columns we want
        display_runs = format_datetime_column(latest_runs, "created_at")
        
        display_cols = ["model_name"] + METRIC_KEYS + ["created_at"]
        available_cols = [col for col in display_cols if col in display_runs.columns]
        st.dataframe(display_runs[available_cols], use_container_width=True, hide_index=True)

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
            
            # Format created_at for display and select only the columns we want
            history_display = model_data.sort_values("created_at", ascending=False).head(10).copy()
            history_display = format_datetime_column(history_display, "created_at")
            
            display_cols = ["model_name"] + METRIC_KEYS + ["created_at"]
            available_cols = [col for col in display_cols if col in history_display.columns]
            st.dataframe(
                history_display[available_cols],
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
            
            # Format deployed_at for display
            deploy_display = format_datetime_column(df_deploy, "deployed_at")
            
            st.dataframe(deploy_display, use_container_width=True, hide_index=True)

            st.markdown("#### Deployment History Timeline")
            try:
                # Prepare data for timeline table
                timeline_df = df_deploy.copy()
                timeline_df = timeline_df.sort_values("deployed_at", ascending=True)
                
                # Mark active deployment
                timeline_df["is_active"] = timeline_df["traffic_split"] > 0.0
                
                # Format deployment date for display
                timeline_df["deployment_date"] = timeline_df["deployed_at"].apply(
                    lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else ""
                )
                
                # Get unique model names (columns) and deployment dates (rows)
                unique_models = sorted(timeline_df["model_name"].dropna().unique())
                unique_dates = timeline_df["deployment_date"].unique()
                
                if len(unique_models) > 0 and len(unique_dates) > 0:
                    # Create pivot table: rows = dates, columns = models
                    timeline_table = pd.DataFrame(index=unique_dates, columns=unique_models)
                    timeline_table = timeline_table.fillna("")  # Fill with empty strings
                    
                    # Store which cells are active for styling
                    active_cells = {}
                    
                    # Fill in the table with deployment status
                    for idx, row in timeline_df.iterrows():
                        date_str = row["deployment_date"]
                        model_name = row["model_name"]
                        if pd.notna(model_name) and date_str in timeline_table.index:
                            if row["is_active"]:
                                timeline_table.at[date_str, model_name] = "✓ Active"
                                active_cells[(date_str, model_name)] = True
                            else:
                                timeline_table.at[date_str, model_name] = "Inactive"
                                active_cells[(date_str, model_name)] = False
                    
                    # Reset index to make deployment_date a column
                    timeline_table = timeline_table.reset_index()
                    timeline_table.columns.name = None
                    timeline_table = timeline_table.rename(columns={"index": "Deployment Date"})
                    
                    # Create HTML table with green background for active cells
                    html = "<table style='width:100%; border-collapse: collapse;'>"
                    
                    # Header row
                    html += "<tr style='background-color: #f0f0f0; font-weight: bold;'>"
                    for col in timeline_table.columns:
                        html += f"<th style='padding: 8px; border: 1px solid #ddd; text-align: left;'>{col}</th>"
                    html += "</tr>"
                    
                    # Data rows
                    for idx, row in timeline_table.iterrows():
                        html += "<tr>"
                        for col in timeline_table.columns:
                            cell_value = row[col] if pd.notna(row[col]) else ""
                            date_str = row["Deployment Date"]
                            
                            # Style based on active status
                            if col == "Deployment Date":
                                html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{cell_value}</td>"
                            elif (date_str, col) in active_cells and active_cells[(date_str, col)]:
                                # Green background for active
                                html += f"<td style='padding: 8px; border: 1px solid #ddd; background-color: #28a745; color: white; font-weight: bold; text-align: center;'>{cell_value}</td>"
                            elif cell_value == "Inactive":
                                # Light gray for inactive
                                html += f"<td style='padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa; color: #6c757d; text-align: center;'>{cell_value}</td>"
                            else:
                                # Empty cell
                                html += f"<td style='padding: 8px; border: 1px solid #ddd;'></td>"
                        html += "</tr>"
                    
                    html += "</table>"
                    
                    # Display the HTML table
                    st.markdown(html, unsafe_allow_html=True)
                    
                    # Add summary information
                    active_count = timeline_df["is_active"].sum()
                    total_count = len(timeline_df)
                    st.caption(f"📊 Total deployments: {total_count} | 🟢 Active: {active_count} | ⚪ Inactive: {total_count - active_count}")
                else:
                    st.info("No deployment data available for timeline table.")
                
            except Exception as e:
                st.warning(f"Unable to generate timeline table: {str(e)}")
                st.error(f"Error details: {str(e)}")
