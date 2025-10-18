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
        # handle known server or empty-data conditions gracefully
        if resp.status_code == 204 or "no new" in resp.text.lower():
            print(f"ℹ️ No new data found for params: {params}. Skipping.")
            raise AirflowSkipException("No new data.")
        elif resp.status_code == 500:
            print(f"⚠️ Server error for {params} (possibly no data or file). Skipping.")
            raise AirflowSkipException("Server-side issue (500).")
        elif resp.status_code >= 400:
            print(f"❌ Request failed with {resp.status_code}: {resp.text}")
            resp.raise_for_status()

        # Try returning JSON, fallback to raw text
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
    }
)
def fred_full_pipeline():
    """
    Full daily FRED pipeline:
    1. Fetch incremental data for each FRED series → GCS (raw-fetch-fred-append)
    2. Append those files into BigQuery (raw-upload-fred-append)
    Gracefully skips if no new data or temporary 500 errors occur.
    """

    CONFIG_PATH = "/usr/local/airflow/include/config/fred_series.yaml"
    with open(CONFIG_PATH, "r") as f:
        series_list = yaml.safe_load(f)["series"]

    @task
    def extract(series_id: str) -> dict:
        """Call Cloud Function to fetch incremental FRED data and save to GCS."""
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw-fetch-fred-append"
        ctx = get_current_context()
        payload = {"series_id": series_id, "run_id": ctx["dag_run"].run_id}

        try:
            resp = invoke_function(url, params=payload)
            print(f"✅ Extracted {series_id}: {resp}")
            return {"series_id": series_id}
        except AirflowSkipException as e:
            print(f"ℹ️ Skipping extract for {series_id}: {e}")
            raise
        except Exception as e:
            print(f"❌ Extraction failed for {series_id}: {e}")
            raise

    @task
    def load_to_bq(extract_payload: dict):
        """Call Cloud Function to append GCS CSVs into BigQuery."""
        series_id = extract_payload["series_id"]
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw-upload-fred-append"

        try:
            resp = invoke_function(url, params={"series_id": series_id})
            print(f"✅ Loaded {series_id} → BigQuery: {resp}")
            return resp
        except AirflowSkipException as e:
            print(f"ℹ️ Skipping load for {series_id}: {e}")
            raise
        except Exception as e:
            print(f"❌ Load failed for {series_id}: {e}")
            raise

    # Dynamically expand over series list
    extract_results = extract.expand(series_id=series_list)
    load_to_bq.expand(extract_payload=extract_results)


fred_full_pipeline()
