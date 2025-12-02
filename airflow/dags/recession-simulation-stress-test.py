from datetime import datetime
from airflow.sdk import dag, task
from pathlib import Path
import utils
from jinja2 import Template
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import os

# Base directory setup
BASE_DIR = Path(os.environ.get("AIRFLOW_HOME", "/usr/local/airflow"))
SQL_DIR = BASE_DIR / "include" / "sql"

# DAG metadata
dag_metadata = {
    "model_id": "recession-simulation",
    "name": "Recession Stress Test Simulation",
    "business_problem": "Simulate macroeconomic downturn and project credit delinquency rates",
    "ticket_number": "BA882-21",
    "owner": "risk_analytics_team"
}

@dag(
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["stress-test", "simulation", "credit-risk"]
)
def simulate_recession_scenario():

    # Pull data from landing layer
    @task
    def extract_recent_data():
        """Pull latest macroeconomic data from the landing layer"""
        query = """
        SELECT *
        FROM landing.fact_macro_indicators_monthly
        ORDER BY date DESC
        LIMIT 24
        """
        data = utils.run_sql(query)
        columns = [col[0] for col in utils.run_sql("SELECT column_name FROM information_schema.columns WHERE table_name = 'fact_macro_indicators_monthly'")]
        df = pd.DataFrame(data, columns=columns)
        print(f"Extracted {df.shape[0]} rows, {df.shape[1]} columns from landing layer.")
        return df.to_json(orient="records")

    # Simulate 6 months of future data
    @task
    def simulate_recession(df_json):
        df = pd.read_json(df_json)

        # Identify date range
        last_date = pd.to_datetime(df["date"].max())
        new_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=6, freq="MS")

        # Create base simulation frame
        sim_df = pd.DataFrame({"date": new_dates})

        # Copy last known values as baseline
        latest = df.iloc[-1:].copy().reset_index(drop=True)
        baseline = pd.concat([latest] * 6, ignore_index=True)
        baseline["date"] = new_dates

        # Apply recession scenario adjustments (hardcoded for now)
        baseline["UNRATE"] = baseline["UNRATE"] + 4.0            # +4 percentage points
        baseline["GDPC1"] = baseline["GDPC1"] * 0.97             # -3% quarter-over-quarter
        baseline["USREC"] = 1                                    # active recession flag

        # Impute other independent variables
        macro_features = [c for c in baseline.columns if c not in ["date", "UNRATE", "GDPC1", "USREC", "DRCCLACBS"]]
        df_all = pd.concat([df[macro_features], baseline[macro_features]], ignore_index=True)

        # Iterative Imputer for time-series consistency
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
        imputed = imputer.fit_transform(df_all)
        df_imputed = pd.DataFrame(imputed, columns=macro_features)
        baseline[macro_features] = df_imputed.iloc[-6:].values

        # Return simulated dataset
        print(f"Simulated recession scenario for {len(baseline)} future months.")
        return baseline.to_json(orient="records")

    # Placeholder for future model inference
    @task
    def placeholder_predict(sim_df_json):
        df = pd.read_json(sim_df_json)
        # Simulate delinquency prediction placeholder (until ML model integration)
        df["predicted_delinquency_rate"] = np.nan  # Placeholder for model output
        print("Created placeholder for predicted delinquency rates.")
        return df.to_json(orient="records")

    # Store results into gold layer
    @task
    def load_to_gold(sim_results_json):
        sim_df = pd.read_json(sim_results_json)

        # Create gold layer table if not exists
        create_sql = """
        CREATE TABLE IF NOT EXISTS gold.fact_recession_scenario_predictions (
            date DATE,
            UNRATE FLOAT64,
            GDPC1 FLOAT64,
            USREC INT64,
            predicted_delinquency_rate FLOAT64,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        utils.run_execute(create_sql)

        # Insert results
        insert_template = Template("""
        INSERT INTO gold.fact_recession_scenario_predictions (date, UNRATE, GDPC1, USREC, predicted_delinquency_rate)
        VALUES
        {% for i, row in df.iterrows() -%}
            ('{{ row.date.date() }}', {{ row.UNRATE }}, {{ row.GDPC1 }}, {{ row.USREC }}, {{ "NULL" if pd.isna(row.predicted_delinquency_rate) else row.predicted_delinquency_rate }}){% if not loop.last %},{% endif %}
        {% endfor %}
        """)
        insert_sql = insert_template.render(df=sim_df, pd=pd)
        utils.run_execute(insert_sql)
        print(f"Inserted {len(sim_df)} simulated rows into gold layer.")

    # Flow of tasks
    raw_data = extract_recent_data()
    simulated_data = simulate_recession(raw_data)
    predictions = placeholder_predict(simulated_data)
    load_to_gold(predictions)

simulate_recession_scenario()
