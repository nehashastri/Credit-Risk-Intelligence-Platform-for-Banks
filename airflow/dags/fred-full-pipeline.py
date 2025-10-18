from airflow.decorators import dag, task
from datetime import datetime, timedelta
from airflow.operators.python import get_current_context
from airflow.exceptions import AirflowSkipException
import requests
import yaml
import os
import json

def invoke_function(url, params={}):
    """Utility to call a Cloud Function safely."""
    try:
        resp = requests.get(url, params=params)

        # Gracefully handle "no new data" or server messages
        if resp.status_code == 204 or "no new" in resp.text.lower():
            print(f"ℹ️ No new data found for params: {params}. Skipping.")
            raise AirflowSkipException("No new data.")
        elif resp.status_code == 500:
            print(f"⚠️ Server error for {params}. Skipping.")
            raise AirflowSkipException("Server-side issue (500).")
        elif resp.status_code >= 400:
            print(f"❌ Request failed with {resp.status_code}: {resp.text}")
            resp.raise_for_status()

        try:
            return resp.json()
        except json.JSONDecodeError:
            return {"response_text": resp.text}
    except AirflowSkipException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error while calling {url}: {e}")
        raise


@dag(
    schedule="@daily",
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["raw", "fred", "append", "bq"],
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5)
    },
    max_active_runs=1
)
def fred_full_pipeline():
    """
    Full daily FRED pipeline:
      1. Fetch incremental CSVs from FRED → GCS (raw-fetch-fred-append)
      2. Append new CSVs into BigQuery raw tables (raw-upload-fred-append)
      3. Trigger landing-load-fred *only if* new data was appended
    """

    CONFIG_PATH = "/usr/local/airflow/include/config/fred_series.yaml"
    with open(CONFIG_PATH, "r") as f:
        series_list = yaml.safe_load(f)["series"]

    @task
    def extract(series_id: str) -> dict:
        """Fetch new FRED data to GCS."""
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw-fetch-fred-append"
        ctx = get_current_context()
        payload = {"series_id": series_id, "run_id": ctx["dag_run"].run_id}

        try:
            resp = invoke_function(url, params=payload)
            print(f"✅ Extracted {series_id}: {resp}")
            return {"series_id": series_id, "new_data": True}
        except AirflowSkipException as e:
            print(f"ℹ️ Skipping extract for {series_id}: {e}")
            return {"series_id": series_id, "new_data": False}
        except Exception as e:
            print(f"❌ Extraction failed for {series_id}: {e}")
            raise

    @task
    def load_to_bq(extract_payload: dict):
        """Append CSVs into BigQuery raw tables."""
        series_id = extract_payload["series_id"]
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw-upload-fred-append"

        # Skip if no new data fetched for this series
        if not extract_payload.get("new_data"):
            print(f"ℹ️ No new data for {series_id}. Skipping load.")
            raise AirflowSkipException("No new data for this series.")

        try:
            resp = invoke_function(url, params={"series_id": series_id})
            print(f"✅ Loaded {series_id} → BigQuery: {resp}")
            return {"series_id": series_id, "loaded": True}
        except AirflowSkipException as e:
            print(f"ℹ️ Skipping load for {series_id}: {e}")
            return {"series_id": series_id, "loaded": False}
        except Exception as e:
            print(f"❌ Load failed for {series_id}: {e}")
            raise

    @task(trigger_rule="all_done")
    def trigger_landing(load_results: list):
        """
        Invoke 'landing-load-fred' Cloud Function *only if*
        any raw tables were updated.
        """
        # Check if any load step returned new data
        updated = any(r.get("loaded") for r in load_results if isinstance(r, dict))

        if not updated:
            print("ℹ️ No new raw data appended — skipping landing update.")
            raise AirflowSkipException("No raw updates → skip landing.")

        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/landing-load-fred"
        ctx = get_current_context()
        payload = {"run_id": ctx["dag_run"].run_id}

        try:
            resp = invoke_function(url, params=payload)
            print("✅ Triggered landing-load-fred successfully:", resp)
            return {"status": "success", "response": resp}
        except Exception as e:
            print("❌ Failed to trigger landing-load-fred:", e)
            raise

    # Pipeline structure
    extract_results = extract.expand(series_id=series_list)
    load_results = load_to_bq.expand(extract_payload=extract_results)
    trigger_landing(load_results)

fred_full_pipeline()
