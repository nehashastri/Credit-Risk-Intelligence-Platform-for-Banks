# dags/credit-risk-pipeline.py
from airflow.decorators import dag, task
from datetime import datetime, timedelta
from airflow.operators.python import get_current_context
from airflow.exceptions import AirflowSkipException
import requests, yaml, json, os

# --------------------------------------------------
# Utility: Safe Cloud Function invoker
# --------------------------------------------------
def invoke_function(url, params={}, method="GET"):
    """Unified Cloud Function invoker with error handling."""
    try:
        if method.upper() == "POST":
            resp = requests.post(url, json=params)
        else:
            resp = requests.get(url, params=params)

        if resp.status_code == 204 or "no new" in resp.text.lower():
            raise AirflowSkipException("No new data.")
        elif resp.status_code == 500:
            raise AirflowSkipException("Server 500 error.")
        elif resp.status_code >= 400:
            print(f"❌ Request failed ({resp.status_code}): {resp.text}")
            resp.raise_for_status()

        try:
            return resp.json()
        except json.JSONDecodeError:
            return {"text": resp.text}
    except AirflowSkipException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error while calling {url}: {e}")
        raise

# --------------------------------------------------
# DAG Definition
# --------------------------------------------------
@dag(
    schedule="@daily",
    start_date=datetime(2025, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["credit-risk", "fred", "yfinance", "ml-dataset"],
)
def credit_risk_pipeline():
    """
    Combined Credit Risk Pipeline:
    1️⃣ FRED incremental pipeline (fetch → upload → landing)
    2️⃣ YFinance incremental pipeline (fetch → upload → landing)
    3️⃣ Invoke create_ml_dataset Cloud Function
    """

    # --------------------------------------------------
    # Config files
    # --------------------------------------------------
    FRED_CONFIG = "/usr/local/airflow/include/config/fred_series.yaml"
    YFIN_CONFIG = "/usr/local/airflow/include/config/yfinance_tickers.yaml"

    with open(FRED_CONFIG, "r") as f:
        fred_series = yaml.safe_load(f)["series"]

    with open(YFIN_CONFIG, "r") as f:
        yfinance_tickers = yaml.safe_load(f)["tickers"]

    # --------------------------------------------------
    # FRED Extract & Load Tasks
    # --------------------------------------------------
    @task
    def extract_fred(series_id: str) -> dict:
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw-fetch-fred-append"
        ctx = get_current_context()
        payload = {"series_id": series_id, "run_id": ctx["dag_run"].run_id}
        try:
            resp = invoke_function(url, params=payload)
            return {"series_id": series_id, "new_data": True}
        except AirflowSkipException:
            return {"series_id": series_id, "new_data": False}

    @task
    def load_fred_to_bq(payload: dict):
        series_id = payload["series_id"]
        if not payload.get("new_data"):
            raise AirflowSkipException("No new FRED data.")
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw_upload_fred_append"
        resp = invoke_function(url, params={"series_id": series_id})
        return {"series_id": series_id, "loaded": True, "resp": resp}

    @task(trigger_rule="all_done")
    def load_fred_landing(results: list):
        if not any(r.get("loaded") for r in results):
            raise AirflowSkipException("No FRED updates → skip landing.")
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/landing-load-fred"
        return invoke_function(url)

    # --------------------------------------------------
    # YFinance Extract & Load Tasks
    # --------------------------------------------------
    @task
    def extract_yfinance(ticker: str) -> dict:
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw_fetch_yfinance_append"
        ctx = get_current_context()
        payload = {"ticker": ticker, "run_id": ctx["dag_run"].run_id}
        resp = invoke_function(url, params=payload)
        return resp

    @task
    def load_yfinance_to_bq(payload: dict) -> dict:
        ticker = payload.get("ticker")
        if not ticker or payload.get("status") in ["skipped", "failed"] or "no_data" in str(payload).lower():
            raise AirflowSkipException(f"No data for {ticker}")
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw_upload_yfinance_append"
        resp = invoke_function(url, params={"ticker": ticker})
        return {"ticker": ticker, "status": "success", "resp": resp}

    @task(trigger_rule="all_done")
    def load_yfinance_landing(results: list):
        if not any(r.get("status") == "success" for r in results):
            raise AirflowSkipException("No YFinance updates → skip landing.")
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/landing_load_yfinance_append"
        return invoke_function(url)

    # --------------------------------------------------
    # Final Step: Trigger ML Dataset Function
    # --------------------------------------------------
    @task(trigger_rule="all_done")
    def create_ml_dataset():
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/create_ml_dataset"
        print("🚀 Triggering ML dataset creation Cloud Function...")
        resp = invoke_function(url)
        print(f"✅ ML dataset creation response: {resp}")
        return resp

    # --------------------------------------------------
    # Task Dependencies
    # --------------------------------------------------
    fred_extracts = extract_fred.expand(series_id=fred_series)
    fred_loads = load_fred_to_bq.expand(payload=fred_extracts)
    fred_landing = load_fred_landing(fred_loads)

    yf_extracts = extract_yfinance.expand(ticker=yfinance_tickers)
    yf_loads = load_yfinance_to_bq.expand(payload=yf_extracts)
    yf_landing = load_yfinance_landing(yf_loads)

    # Run ML dataset creation after both pipelines complete
    create_ml_dataset().set_upstream([fred_landing, yf_landing])


# Instantiate DAG
credit_risk_pipeline()
