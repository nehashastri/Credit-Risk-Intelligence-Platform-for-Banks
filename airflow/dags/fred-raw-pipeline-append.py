from airflow.decorators import dag, task
from datetime import datetime
from airflow.operators.python import get_current_context
import requests
import yaml
import os

def invoke_function(url, params={}):
    """Utility to invoke a Cloud Function endpoint and return its JSON response."""
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

@dag(
    schedule="@daily",
    start_date=datetime(2025, 10, 1),
    catchup=False,  # run only current and future dates (no backfill)
    max_active_runs=1,
    tags=["raw", "fred", "append", "incremental"]
)
def fred_raw_pipeline_append():
    """
    DAG for incremental FRED → GCS → BigQuery ingestion.
    Uses 'raw-fetch-fred-append' Cloud Function to fetch only new data
    and 'raw-upload-fred' to load it into BigQuery.
    """

    CONFIG_PATH = "/usr/local/airflow/include/config/fred_series.yaml"

    # Read list of FRED series to fetch
    with open(CONFIG_PATH, "r") as f:
        series_list = yaml.safe_load(f)["series"]

    @task
    def extract(series_id: str) -> dict:
        """Call incremental Cloud Function to fetch only new data and upload to GCS."""
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw-fetch-fred-append"
        ctx = get_current_context()
        payload = {"series_id": series_id, "run_id": ctx["dag_run"].run_id}
        resp = invoke_function(url, params=payload)
        return resp

    @task
    def load(extract_payload: dict) -> dict:
        """Call Cloud Function to load newly created GCS CSV into BigQuery."""
        if not extract_payload.get("gcs_path"):
            print(f"No new data to load for {extract_payload.get('series_id')}")
            return {"status": "skipped"}
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw-upload-fred"
        series_id = extract_payload["series_id"]
        resp = invoke_function(url, params={"series_id": series_id})
        return resp

    # Dynamically map tasks over each series
    extract_results = extract.expand(series_id=series_list)
    load.expand(extract_payload=extract_results)

fred_raw_pipeline_append()
