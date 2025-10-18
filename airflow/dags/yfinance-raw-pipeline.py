from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from datetime import datetime
import requests

def invoke_function(url, params={}):
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

@dag(
    schedule="@daily",
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["raw", "yfinance", "ingest"]
)
def yfinance_raw_pipeline():

    @task
    def extract() -> dict:
        # Get the DAG run context here, inside the task
        ctx = get_current_context()
        run_id = ctx["dag_run"].run_id

        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw-fetch-yfinance"
        payload = {"run_id": run_id}  # Cloud Function handles all tickers internally
        resp = invoke_function(url, params=payload)
        return resp

    # Call task
    extract()

# Instantiate the DAG
yfinance_raw_pipeline()
