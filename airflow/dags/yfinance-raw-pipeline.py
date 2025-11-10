from airflow.decorators import dag, task
from datetime import datetime
from airflow.operators.python import get_current_context
import requests
import yaml
import os

# ------------------------------------------------
# Helper function to call Cloud Functions
# ------------------------------------------------
def invoke_function(url, params={}):
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

# ------------------------------------------------
# DAG Definition
# ------------------------------------------------
@dag(
    schedule="@daily",
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["raw", "yfinance", "ingest"]
)
def yfinance_raw_pipeline():
    """
    DAG to fetch daily historical data for YFinance tickers (1990-present)
    and load them into BigQuery via Cloud Functions.
    """

    CONFIG_PATH = "/usr/local/airflow/include/config/yfinance_tickers.yaml"

    # Load ticker list from YAML
    with open(CONFIG_PATH, "r") as f:
        tickers = yaml.safe_load(f)["tickers"]

    @task
    def extract(ticker: str) -> dict:
        """Call Cloud Function to fetch yfinance data and store CSV in GCS."""
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw_fetch_yfinance"
        ctx = get_current_context()
        payload = {"series_id": ticker, "run_id": ctx["dag_run"].run_id}
        resp = invoke_function(url, params=payload)
        print(f"Fetched {ticker}: {resp}")
        return resp

    @task
    def load(extract_payload: dict) -> dict:
        """Call Cloud Function to load CSV from GCS into BigQuery."""
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw-upload-yfinance"
        series_id = extract_payload["series_id"]
        resp = invoke_function(url, params={"series_id": series_id})
        print(f"Loaded {series_id}: {resp}")
        return resp

    # Dynamic task mapping
    extract_results = extract.expand(ticker=tickers)
    load.expand(extract_payload=extract_results)


# Instantiate the DAG
yfinance_raw_pipeline()
