from datetime import timedelta
import os
import yaml
import requests
import pendulum

from airflow import DAG
from airflow.models import Variable
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

# Logging timezone
ET = pendulum.timezone("America/New_York")

# --------------------------------------------------
# Helper: locate the YAML query configuration file
# --------------------------------------------------
def _queries_yaml_path() -> str:
    """
    Returns the absolute path to the queries.yaml configuration file.
    Adjusted for the directory structure:
    airflow/dags/include/config/queries.yaml
    """
    here = os.path.dirname(os.path.abspath(__file__))  # current: airflow/dags/
    return os.path.normpath(os.path.join(here, "include", "config", "queries.yaml"))

# --------------------------------------------------
# Helper: load query presets from YAML
# --------------------------------------------------
def _load_presets_from_yaml(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    defaults = data.get("defaults", {}) or {}
    presets  = data.get("presets", []) or []

    # Allow Airflow Variables or environment variables to override defaults
    raw_days = Variable.get("RAW_DAYS", default_var=os.getenv("RAW_DAYS"))
    raw_max  = Variable.get("RAW_MAX_RESULTS", default_var=os.getenv("RAW_MAX_RESULTS"))
    if raw_days is not None:
        defaults["days"] = int(raw_days)
    if raw_max is not None:
        defaults["max_results"] = int(raw_max)

    merged = []
    for p in presets:
        merged.append({
            "topic": p["topic"],
            "query": p["query"],
            "days": int(p.get("days", defaults.get("days", 3))),
            "max_results": int(p.get("max_results", defaults.get("max_results", 20))),
        })
    return defaults, merged

# --------------------------------------------------
# Cloud Run Invocation Task
# --------------------------------------------------
def invoke_raw_fetch_news():
    # Use Airflow Variable or fallback to default URL
    url = (Variable.get("RAW_FETCH_NEWS_URL", default_var=None)
           or os.getenv("RAW_FETCH_NEWS_URL")
           or "https://raw-fetch-news-265141170939.us-central1.run.app")
    print(f"[invoke_raw_fetch_news] Cloud Run URL: {url}")

    defaults, presets = _load_presets_from_yaml(_queries_yaml_path())
    payload = {
        "defaults": {
            "days": int(defaults.get("days", 3)),
            "max_results": int(defaults.get("max_results", 20)),
        },
        "presets": presets,
    }

    # Increased timeout to handle long Cloud Run execution
    resp = requests.post(url, json=payload, timeout=300)
    if resp.status_code not in (200, 204):
        raise RuntimeError(f"[raw] CF call failed: {resp.status_code} {resp.text}")

# --------------------------------------------------
# DAG Definition
# --------------------------------------------------
with DAG(
    dag_id="news_raw_pipeline",
    description="Fetch news data into BigQuery raw layer via Cloud Run",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="15 8 * * *",  # Runs daily at 08:15 UTC
    catchup=False,
    is_paused_upon_creation=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["news", "raw"],
) as dag:

    # Task 1: Invoke the Cloud Run service
    call_cf = PythonOperator(
        task_id="invoke_raw_fetch_news",
        python_callable=invoke_raw_fetch_news,
        execution_timeout=timedelta(minutes=6),
    )

    # Task 2: Trigger the landing pipeline after raw completes successfully
    trigger_landing = TriggerDagRunOperator(
        task_id="trigger_news_landing_pipeline",
        trigger_dag_id="news_landing_pipeline",
        reset_dag_run=True,  # ensures reruns overwrite old runs
        wait_for_completion=False,
    )

    call_cf >> trigger_landing