# streamlit/pages/scenario.py

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import date, timedelta

# ----------------------------------------------------------------------
# Cloud Function endpoint for "precomputed mean scenario" prediction
#   - This calls ml-predict-model
#   - It reads BigQuery table fact_all_indicators_weekly_llm_scenario_mean
# ----------------------------------------------------------------------
PREDICT_ENDPOINT = st.secrets.get(
    "PREDICT_ENDPOINT",
    "https://us-central1-pipeline-882-team-project.cloudfunctions.net/ml-predict-model",
)

# ----------------------------------------------------------------------
# Cloud Function endpoint for real-time "What-if" scenario generation
#   - This calls ml-generate-scenario
#   - It uses LLM + deployed model to run prediction immediately
# ----------------------------------------------------------------------
SCENARIO_GEN_ENDPOINT = st.secrets.get(
    "SCENARIO_GEN_ENDPOINT",
    "https://us-central1-pipeline-882-team-project.cloudfunctions.net/ml-generate-scenario",
)


def show_scenario_page():
    # =======================================================
    # Page header and description
    # =======================================================
    st.title("📈 Scenario-based Delinquency Forecast")
    st.caption("Run delinquency predictions from both precomputed and custom scenarios")

    # =======================================================
    # Section A — Precomputed LLM mean scenario
    # =======================================================
    st.markdown("### A. Precomputed LLM Scenario (Mean Table)")
    st.markdown("#### 1️⃣ Select forecast range")

    today = date.today()
    default_start = today
    default_end = today + timedelta(weeks=8)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date", value=default_start, key="mean_start_date"
        )
    with col2:
        end_date = st.date_input("End date", value=default_end, key="mean_end_date")

    st.markdown("#### 2️⃣ Run prediction from mean table")

    if st.button("🚀 Run Forecast from Mean Scenario", key="btn_mean_scenario"):
        with st.spinner("Calling model prediction API using mean scenario..."):
            try:
                params = {
                    "source": "scenario_mean",
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "limit": 200,
                }

                resp = requests.get(PREDICT_ENDPOINT, params=params, timeout=60)

                if resp.status_code != 200:
                    st.error(f"API error {resp.status_code}: {resp.text}")
                    return

                data = resp.json()
                preds = data.get("predictions", [])

                if not preds:
                    st.warning("No predictions returned from the API.")
                    return

                df = pd.DataFrame(preds)
                st.success(f"✅ Received {len(df)} predictions")

                # Line chart for predicted delinquency rate
                if "predicted_delinquency_rate" in df.columns:
                    fig = px.line(
                        df,
                        x="date",
                        y="predicted_delinquency_rate",
                        title="Predicted Delinquency Rate (Precomputed Scenario Mean)",
                        markers=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Raw results table (hide actual_delinquency_rate, week, year)
                st.markdown("#### 🔍 Raw Prediction Data")

                drop_cols = ["actual_delinquency_rate", "week", "year"]
                display_cols = [c for c in df.columns if c not in drop_cols]
                df_display = df[display_cols].copy()

                st.dataframe(df_display, use_container_width=True)

            except Exception as e:
                st.error(f"Error calling prediction endpoint: {e}")

    st.markdown("---")

    # =======================================================
    # Section B — Real-time What-if Scenario
    # =======================================================
    st.markdown("### B. Real-time What-if Scenario (Gemini)")

    st.markdown(
        """
Describe a scenario in natural language, and we will:

1. Use **Gemini** to generate a forward macro scenario (8 weeks), and  
2. Run the **deployed delinquency model** on the generated scenario.
"""
    )

    st.markdown("#### 3️⃣ Describe the custom scenario")

    col_left, col_right = st.columns([2, 1])

    with col_right:
        interest_rate = st.slider("Interest rate (%)", 0.0, 10.0, 3.5, 0.25)
        unemployment_delta = st.slider(
            "Unemployment change (pp)", -3.0, 5.0, 1.0, 0.5
        )

    with col_left:
        default_scenario_text = (
            f"If the interest rate is set to {interest_rate:.2f}% "
            f"and unemployment increases by {unemployment_delta:.1f} percentage points, "
            "how should macro indicators evolve over the next 8 weeks?"
        )
        scenario_text = st.text_area(
            "Scenario description",
            value=default_scenario_text,
            height=120,
        )

    horizon_weeks = st.number_input(
        "Prediction horizon (weeks)",
        min_value=4,
        max_value=16,
        value=8,
        step=1,
    )

    st.markdown("#### 4️⃣ Generate scenario and run model")

    if st.button("🤖 Generate Scenario & Predict", key="btn_custom_scenario"):
        with st.spinner("Calling OpenAI API + deployed model..."):
            try:
                payload = {
                    "scenario_text": scenario_text,
                    "horizon_weeks": int(horizon_weeks),
                }

                resp = requests.post(
                    SCENARIO_GEN_ENDPOINT, json=payload, timeout=90
                )

                if resp.status_code != 200:
                    st.error(f"Scenario API error {resp.status_code}: {resp.text}")
                    return

                data = resp.json()
                preds = data.get("predictions", [])
                scenario_rows = data.get("scenario_rows", [])

                if not preds:
                    st.warning("No predictions returned.")
                    return

                df_pred = pd.DataFrame(preds)

                st.success(
                    f"✅ Scenario generated and predictions produced: {len(df_pred)} rows "
                    f"(deployment={data.get('deployment_id')})"
                )

                # Line chart for predicted delinquency rate
                if "predicted_delinquency_rate" in df_pred.columns:
                    fig = px.line(
                        df_pred,
                        x="date",
                        y="predicted_delinquency_rate",
                        title="Predicted Delinquency Rate (Custom Scenario)",
                        markers=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Raw prediction table (only the columns we want)
                st.markdown("#### 🔍 Raw Prediction Data")

                base_display_cols = ["date", "predicted_delinquency_rate"]
                display_cols = [c for c in base_display_cols if c in df_pred.columns]

                df_display = df_pred[display_cols].copy()

                if "date" in df_display.columns:
                    df_display["date"] = pd.to_datetime(
                        df_display["date"]
                    ).dt.strftime("%Y-%m-%d")

                st.dataframe(df_display, use_container_width=True)

                # Show LLM-generated macro scenario used as model input
                if scenario_rows:
                    st.markdown("#### 🧠 LLM-generated Macro Scenario (Model Input)")
                    df_scenario = pd.DataFrame(scenario_rows)
                    st.dataframe(df_scenario, use_container_width=True)

            except Exception as e:
                st.error(f"Error calling scenario generation endpoint: {e}")