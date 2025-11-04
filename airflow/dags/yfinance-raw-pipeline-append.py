# ===============================================
# DAG: yfinance_raw_pipeline_append
# Fetch daily YFinance data → Upload to GCS → Append to BigQuery → Load into Landing
# ===============================================

from airflow.sdk import dag, task, get_current_context
from datetime import datetime
import requests
import yaml
import os

# --------------------------------------------------
# Utility Function
# --------------------------------------------------
def invoke_function(url, params={}, method="GET"):
    """Utility to call a Cloud Function endpoint and return its JSON response."""
    try:
        if method.upper() == "POST":
            resp = requests.post(url, json=params)
        else:
            resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling {url}: {e}")
        return {"error": str(e), "status": "failed"}


# --------------------------------------------------
# DAG Definition
# --------------------------------------------------
@dag(
    schedule="@daily",
    start_date=datetime(2025, 10, 1),
    catchup=False,
    max_active_runs=1,
    tags=["raw", "yfinance", "append", "landing", "incremental"],
)
def yfinance_raw_pipeline_append():
    """
    DAG for incremental Yahoo Finance → GCS → BigQuery ingestion → Landing transformation.

    Step 1: Fetch daily data (Cloud Function: raw_fetch_yfinance_append)
    Step 2: Append to BigQuery (Cloud Function: raw_upload_yfinance_append)
    Step 3: Load new data into landing.fact_sector_prices_volumes_append
    """

    CONFIG_PATH = "/usr/local/airflow/include/config/yfinance_tickers.yaml"

    # -------------------------
    # Read tickers from YAML config
    # -------------------------
    with open(CONFIG_PATH, "r") as f:
        tickers = yaml.safe_load(f)["tickers"]

    # -------------------------
    # Task 1: Extract (Fetch + Upload to GCS)
    # -------------------------
    @task
    def extract(ticker: str) -> dict:
        """Fetch the latest YFinance data for a ticker and upload it to GCS."""
        url = (
            "https://us-central1-pipeline-882-team-project.cloudfunctions.net/"
            "raw_fetch_yfinance_append"
        )
        ctx = get_current_context()
        payload = {"ticker": ticker, "run_id": ctx["dag_run"].run_id}

        print(f"🚀 Fetching data for {ticker} ...")
        resp = invoke_function(url, params=payload)
        print(f"📦 Extract result for {ticker}: {resp}")
        return resp

    # -------------------------
    # Task 2: Load (Append to BigQuery)
    # -------------------------
    @task(retries=2, retry_delay=60)
    def load(extract_payload: dict) -> dict:
        """
        Append uploaded CSV into BigQuery (sector_equity_features_2).
        Calls Cloud Function: raw_upload_yfinance_append
        """
        ticker = extract_payload.get("ticker")

        if (
            not extract_payload.get("gcs_path")
            or "no_data" in str(extract_payload).lower()
            or "skipped" in str(extract_payload).lower()
            or "empty" in str(extract_payload).lower()
            or extract_payload.get("status") in ["no_data", "skipped"]
        ):
            print(f"⏩ Skipping load — no valid data for {ticker}")
            return {"ticker": ticker, "status": "skipped"}

        url = (
            "https://us-central1-pipeline-882-team-project.cloudfunctions.net/"
            "raw_upload_yfinance_append"
        )

        print(f"⬆️ Uploading data for {ticker} to BigQuery → sector_equity_features_2")
        resp = invoke_function(url, params={"ticker": ticker})

        if resp.get("error"):
            print(f"❌ Upload failed for {ticker}: {resp['error']}")
            return {"ticker": ticker, "status": "failed", "error": resp["error"]}

        print(f"✅ Load successful for {ticker}: {resp}")
        return {
            "ticker": ticker,
            "status": "success",
            "rows_uploaded": resp.get("rows_uploaded", 0),
            "bq_table": resp.get("bq_table"),
            "gcs_source": resp.get("gcs_source"),
        }

    # -------------------------
    # Task 3: Load into Landing (only if new data exists)
    # -------------------------
    @task
    def load_landing(load_results: list) -> dict:
        """
        Run landing_load_yfinance_append Cloud Function only if any ticker had new data.
        """
        # Check if any ticker actually uploaded new rows
        new_data = any(
            r.get("status") == "success" and r.get("rows_uploaded", 0) > 0
            for r in load_results
            if isinstance(r, dict)
        )

        if not new_data:
            print("⏩ No new data found across tickers — skipping landing load.")
            return {"status": "skipped", "message": "No new data"}

        url = (
            "https://us-central1-pipeline-882-team-project.cloudfunctions.net/"
            "landing_load_yfinance_append"
        )

        print("🚀 Triggering landing transformation → landing.fact_sector_prices_volumes_append ...")
        resp = invoke_function(url)
        print(f"✅ Landing load response: {resp}")
        return resp

    # -------------------------
    # Parallel & Conditional Flow
    # -------------------------
    extract_results = extract.expand(ticker=tickers)
    load_results = load.expand(extract_payload=extract_results)
    load_landing(load_results)

# -------------------------
# Instantiate DAG
# -------------------------
yfinance_raw_pipeline_append()
