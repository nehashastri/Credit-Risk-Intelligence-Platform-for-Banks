from datetime import timedelta
import pendulum
import requests
import os

from airflow import DAG
from airflow.models import Variable
from airflow.providers.standard.operators.python import PythonOperator

ET = pendulum.timezone("America/New_York")
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "pipeline-882-team-project")

# --------------------------------------------------
# Cloud Run Invocation Task
# --------------------------------------------------
def invoke_landing_load_news():
    url = (Variable.get("LANDING_LOAD_NEWS_URL", default_var=None)
           or os.getenv("LANDING_LOAD_NEWS_URL")
           or "https://landing-load-news-265141170939.us-central1.run.app")

    print(f"[invoke_landing_load_news] Cloud Run URL: {url}")

    resp = requests.post(url, json={}, timeout=180)
    if resp.status_code not in (200, 204):
        raise RuntimeError(f"[landing] CF call failed: {resp.status_code} {resp.text}")

# --------------------------------------------------
# DAG Definition
# --------------------------------------------------
with DAG(
    dag_id="news_landing_pipeline",
    description="Load news data from raw → landing and aggregate the last 7 days",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,  # No direct schedule, triggered by raw pipeline
    catchup=False,
    is_paused_upon_creation=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["news", "landing", "bigquery"],
) as dag:

    trigger_landing_load = PythonOperator(
        task_id="invoke_landing_load_news",
        python_callable=invoke_landing_load_news,
        execution_timeout=timedelta(minutes=4),
    )