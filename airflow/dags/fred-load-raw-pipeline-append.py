from airflow.decorators import dag, task
from datetime import datetime
from airflow.exceptions import AirflowSkipException
import requests
import yaml
import json

@dag(
    schedule="@daily",
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["fred", "append", "bq"]
)
def fred_load_raw_pipeline_append():
    CONFIG_PATH = "/usr/local/airflow/include/config/fred_series.yaml"

    @task
    def trigger_load(series_id: str):
        """
        Calls Cloud Function to append new FRED CSVs to BigQuery.
        Gracefully handles cases where no new file exists or Cloud Function returns 500.
        """
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw-upload-fred-append"
        try:
            resp = requests.get(url, params={"series_id": series_id})
            
            # --- Case 1: No new data ---
            if resp.status_code == 204 or "no new" in resp.text.lower():
                print(f"ℹ️ No new data for {series_id}. Skipping append.")
                raise AirflowSkipException(f"No updates for {series_id}.")
            
            # --- Case 2: Server-side issue (likely no files in GCS) ---
            elif resp.status_code == 500:
                print(f"⚠️ Server error for {series_id} (possibly no new files). Skipping.")
                raise AirflowSkipException(f"500 - No new files or internal error for {series_id}.")
            
            # --- Case 3: Any other client error ---
            elif resp.status_code >= 400:
                print(f"❌ Request failed for {series_id} with {resp.status_code}: {resp.text}")
                resp.raise_for_status()
            
            # --- Case 4: Successful append ---
            else:
                try:
                    data = resp.json()
                except json.JSONDecodeError:
                    data = {"message": resp.text}
                print(f"✅ Successfully appended {series_id}: {data}")
                return {"series_id": series_id, "status": "success"}

        except AirflowSkipException:
            raise  # explicitly mark skipped
        except Exception as e:
            print(f"❌ Unexpected error for {series_id}: {e}")
            raise e

    # Load YAML series list
    with open(CONFIG_PATH, "r") as f:
        series_list = yaml.safe_load(f)["series"]

    trigger_load.expand(series_id=series_list)

fred_load_raw_pipeline_append()
