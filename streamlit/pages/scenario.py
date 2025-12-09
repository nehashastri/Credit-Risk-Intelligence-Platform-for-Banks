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
    "https://us-central1-pipeline-882-team-project.cloudfunctions.net/ml-predict-model"
)

# ----------------------------------------------------------------------
# Cloud Function endpoint for real-time "What-if" scenario generation
#   - This calls ml-generate-scenario
#   - It uses Gemini + deployed model to run prediction immediately
# ----------------------------------------------------------------------
SCENARIO_GEN_ENDPOINT = st.secrets.get(
    "SCENARIO_GEN_ENDPOINT",
    "https://us-central1-pipeline-882-team-project.cloudfunctions.net/ml-generate-scenario"
)


def show_scenario_page():
    # -------------------------------------------------------
    # Page header and introductory text
    # -------------------------------------------------------
    st.title("📈 Scenario-based Delinquency Forecast")
    st.caption("Run delinquency predictions from both precomputed and custom scenarios")

    # =======================================================
    # Section A — Use the precomputed LLM mean scenario table
    # =======================================================
    st.markdown("### A. Precomputed LLM Scenario (Mean Table)")
    
    st.info(
        "💡 **Tip:** This section reads from the precomputed scenario table. "
        "If you get a 'No data found' error, try adjusting the date range to include dates where data exists in the table."
    )

    st.markdown("#### 1️⃣ Select forecast range")

    today = date.today()
    # Use a past date as default start to ensure data exists
    default_start = today - timedelta(weeks=1)
    default_end = today + timedelta(weeks=8)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=default_start, key="mean_start_date")
    with col2:
        end_date = st.date_input("End date", value=default_end, key="mean_end_date")
    
    # Validate date range
    if start_date > end_date:
        st.error("⚠️ Start date must be before end date.")
        st.stop()

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

                if resp.status_code == 404:
                    error_data = resp.json() if resp.headers.get('content-type', '').startswith('application/json') else {}
                    error_msg = error_data.get("error", "No data found")
                    st.warning(f"⚠️ {error_msg}")
                    st.info(
                        f"**Troubleshooting tips:**\n"
                        f"- Try adjusting the date range (currently: {start_date} to {end_date})\n"
                        f"- The table `fact_all_indicators_weekly_llm_scenario_mean` may not have data for this date range\n"
                        f"- Check if the table exists and contains data in BigQuery"
                    )
                    return
                elif resp.status_code != 200:
                    error_text = resp.text
                    try:
                        error_data = resp.json()
                        error_msg = error_data.get("error", error_text)
                        error_details = error_data.get("details", "")
                        st.error(f"❌ API error {resp.status_code}: {error_msg}")
                        if error_details:
                            st.error(f"Details: {error_details}")
                    except:
                        st.error(f"❌ API error {resp.status_code}: {error_text}")
                    return

                data = resp.json()
                preds = data.get("predictions", [])

                if not preds:
                    st.warning("No predictions returned from the API.")
                    return

                df = pd.DataFrame(preds)
                st.success(f"✅ Received {len(df)} predictions")

                # Line chart for predicted rate
                if "predicted_delinquency_rate" in df.columns and "date" in df.columns:
                    # Convert date column to datetime if it's a string
                    df_plot = df.copy()
                    if df_plot["date"].dtype == "object":
                        df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")
                    
                    fig = px.line(
                        df_plot,
                        x="date",
                        y="predicted_delinquency_rate",
                        title="Predicted Delinquency Rate (Precomputed Scenario Mean)",
                        markers=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Show raw results
                st.markdown("#### 🔍 Raw Prediction Data")
                st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Error calling prediction endpoint: {e}")

    st.markdown("---")

    # =======================================================
    # Section B — Real-time What-if Scenario (Gemini + model)
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
        unemployment_delta = st.slider("Unemployment change (pp)", -3.0, 5.0, 1.0, 0.5)

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
        with st.spinner("Calling Gemini + deployed model..."):
            try:
                payload = {
                    "scenario_text": scenario_text,
                    "horizon_weeks": int(horizon_weeks),
                }

                resp = requests.post(SCENARIO_GEN_ENDPOINT, json=payload, timeout=90)

                if resp.status_code == 500:
                    error_text = resp.text
                    try:
                        error_data = resp.json()
                        error_msg = error_data.get("error", error_text)
                        error_details = error_data.get("details", "")
                        st.error(f"❌ Server error {resp.status_code}: {error_msg}")
                        if error_details:
                            st.error(f"Details: {error_details}")
                            if "db-dtypes" in error_details.lower():
                                st.info(
                                    "**Note:** This error indicates the Cloud Function needs the `db-dtypes` package. "
                                    "This is a backend dependency issue that needs to be fixed in the Cloud Function deployment. "
                                    "Please check the Cloud Function's requirements.txt file."
                                )
                    except:
                        st.error(f"❌ Server error {resp.status_code}: {error_text}")
                    return
                elif resp.status_code != 200:
                    error_text = resp.text
                    try:
                        error_data = resp.json()
                        error_msg = error_data.get("error", error_text)
                        error_details = error_data.get("details", "")
                        st.error(f"❌ API error {resp.status_code}: {error_msg}")
                        if error_details:
                            st.error(f"Details: {error_details}")
                    except:
                        st.error(f"❌ API error {resp.status_code}: {error_text}")
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

                # Predicted delinquency chart
                if "predicted_delinquency_rate" in df_pred.columns and "date" in df_pred.columns:
                    # Convert date column to datetime if it's a string
                    df_pred_plot = df_pred.copy()
                    if df_pred_plot["date"].dtype == "object":
                        df_pred_plot["date"] = pd.to_datetime(df_pred_plot["date"], errors="coerce")
                    
                    fig = px.line(
                        df_pred_plot,
                        x="date",
                        y="predicted_delinquency_rate",
                        title="Predicted Delinquency Rate (Custom Scenario)",
                        markers=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Raw prediction table
                st.markdown("#### 🔍 Raw Prediction Data (Custom Scenario)")
                st.dataframe(df_pred, use_container_width=True)

                # Show generated macro scenario
                if scenario_rows:
                    st.markdown("#### 🧠 LLM-generated Macro Scenario (Model Input)")
                    df_scenario = pd.DataFrame(scenario_rows)
                    st.dataframe(df_scenario, use_container_width=True)

            except Exception as e:
                st.error(f"Error calling scenario generation endpoint: {e}")