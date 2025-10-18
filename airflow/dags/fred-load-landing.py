from airflow.decorators import dag, task
from datetime import datetime
from airflow.operators.python import get_current_context
import requests
import os

# 👇 Helper function to call the Cloud Function
def invoke_function(url, params={}):
    """Helper to call Cloud Function via HTTP GET."""
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

@dag(
    schedule="@daily",  # runs once per day
    start_date=datetime(2025, 10, 1),
    catchup=False,  # don't backfill,
    default_args={"retries": 0},
    tags=["landing", "fred", "load", "bigquery"]
)
def fred_load_landing_pipeline():
    """
    DAG that loads cleaned FRED data from raw zone into BigQuery landing tables
    by invoking the Cloud Function 'landing-load-fred'.
    """

    # 🔹 Step 1: Define Cloud Function URL
    # (this is the deployed Cloud Function endpoint)
    url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/landing-load-fred"

    # 🔹 Step 2: Define the landing load task
    @task
    def load_landing():
        """
        Invokes the 'landing-load-fred' Cloud Function to populate
        three landing tables in BigQuery.
        """
        ctx = get_current_context()
        payload = {
            "run_id": ctx["dag_run"].run_id,
            "source_dataset": "raw",
            "target_dataset": "landing"
        }

        try:
            resp = invoke_function(url, params=payload)
            print("✅ Successfully loaded landing tables:", resp)
            return {"status": "success", "response": resp}
        except Exception as e:
            print("❌ Failed to load landing tables:", e)
            return {"status": "failed", "error": str(e)}

    # 🔹 Step 3: Run the load task
    load_landing()

fred_load_landing_pipeline()
