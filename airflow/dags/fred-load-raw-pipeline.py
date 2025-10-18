from airflow.decorators import dag, task
from datetime import datetime
from airflow.operators.python import get_current_context
import requests
import yaml
import os

# 👇 Helper function to call the Cloud Function
def invoke_function(url, params={}):
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

@dag(
    schedule="@daily",  # runs once per day
    start_date=datetime(2025, 10, 1),
    catchup=False,  # don't backfill
    tags=["raw", "fred", "load", "bigquery"]
)
def fred_load_raw_pipeline():
    """
    DAG that loads FRED data from GCS into BigQuery using the deployed Cloud Function.
    """

    CONFIG_PATH = "/usr/local/airflow/include/config/fred_series.yaml"

    # 🔹 Step 1: Read all series from YAML
    with open(CONFIG_PATH, "r") as f:
        series_list = yaml.safe_load(f)["series"]

    # 🔹 Step 2: Define the load task
    @task
    def load_to_bq(series_id: str):
        """
        For each series_id, invoke the Cloud Function to load data into BigQuery.
        """
        url = "https://task-r25wwaz52q-uc.a.run.app"  # 👈 your Cloud Function URL
        ctx = get_current_context()
        payload = {"series_id": series_id, "run_id": ctx["dag_run"].run_id}

        try:
            resp = invoke_function(url, params=payload)
            print(f"✅ Successfully loaded {series_id}: {resp}")
            return {"series_id": series_id, "status": "success", "response": resp}
        except Exception as e:
            print(f"❌ Failed to load {series_id}: {e}")
            return {"series_id": series_id, "status": "failed", "error": str(e)}

    # 🔹 Step 3: Map across all FRED series
    load_to_bq.expand(series_id=series_list)

fred_load_raw_pipeline()
