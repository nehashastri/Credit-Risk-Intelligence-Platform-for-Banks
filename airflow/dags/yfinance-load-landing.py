from airflow.decorators import dag, task
from datetime import datetime
from airflow.operators.python import get_current_context
import requests

# 👇 Helper function to call Cloud Function
def invoke_function(url, params={}):
    """Helper to call Cloud Function via HTTP GET."""
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

@dag(
    schedule="@daily",  # runs once per day
    start_date=datetime(2025, 10, 1),
    catchup=False,
    default_args={"retries": 0},
    tags=["landing", "yfinance", "load", "bigquery"]
)
def yfinance_load_landing_pipeline():
    """
    DAG that loads cleaned yfinance data from raw zone into BigQuery landing tables
    by invoking the Cloud Function 'landing-load-yfinance'.
    """

    # 🔹 Step 1: Define Cloud Function URL
    CF_URL = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/landing_load_sector_equity"

    # 🔹 Step 2: Define the landing load task
    @task
    def load_landing():
        """
        Invokes the 'landing-load-yfinance' Cloud Function to populate
        landing tables in BigQuery.
        """
        ctx = get_current_context()
        payload = {
            "run_id": ctx["dag_run"].run_id,
            "source_dataset": "raw",
            "target_dataset": "landing"
        }

        try:
            resp = invoke_function(CF_URL, params=payload)
            print("✅ Successfully loaded yfinance landing tables:", resp)
            return {"status": "success", "response": resp}
        except Exception as e:
            print("❌ Failed to load landing tables:", e)
            return {"status": "failed", "error": str(e)}

    # 🔹 Step 3: Run the load task
    load_landing()

yfinance_load_landing_pipeline()
