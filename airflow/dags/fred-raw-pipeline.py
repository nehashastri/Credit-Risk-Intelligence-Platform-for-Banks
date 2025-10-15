from airflow.decorators import dag, task
from datetime import datetime
from airflow.operators.python import get_current_context
import requests
import yaml
import os

def invoke_function(url, params={}):
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

@dag(
    schedule="@daily",
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["raw", "fred", "ingest"]
)
def fred_raw_pipeline():
    CONFIG_PATH = "/usr/local/airflow/include/config/fred_series.yaml"
    with open(CONFIG_PATH, "r") as f:
        series_list = yaml.safe_load(f)["series"]

    @task
    def extract(series_id: str) -> dict:
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw-fetch-fred"
        ctx = get_current_context()
        payload = {"series_id": series_id, "run_id": ctx["dag_run"].run_id}
        resp = invoke_function(url, params=payload)
        return resp

    @task
    def load(extract_payload: dict) -> dict:
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw-upload-fred"
        series_id = extract_payload["series_id"]
        resp = invoke_function(url, params={"series_id": series_id})
        return resp

    extract_results = extract.expand(series_id=series_list)
    load.expand(extract_payload=extract_results)

fred_raw_pipeline()