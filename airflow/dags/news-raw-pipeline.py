# news-raw-pipeline.py
# Fetches news into the BigQuery RAW layer via Cloud Run.
# After a successful run, it triggers the landing pipeline.

from datetime import timedelta
import os
import yaml
import requests
import pendulum

from airflow import DAG
from airflow.models import Variable
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator


# Optional timezone (useful for logs)
ET = pendulum.timezone("America/New_York")


# ---------------------------------------------------------------------
# Robust locator for queries.yaml (works across bundle/mount layouts)
# ---------------------------------------------------------------------
def _queries_yaml_path() -> str:
    """Return a readable path to queries.yaml, trying several locations.
    Falls back to an Airflow Variable 'QUERIES_YAML' if set (inline content)."""
    here = os.path.dirname(os.path.abspath(__file__))  # .../airflow/dags/

    candidates = [
        # Project layout: airflow/dags/../include/config/queries.yaml
        os.path.normpath(os.path.join(here, "..", "include", "config", "queries.yaml")),
        # If someone placed include under dags inadvertently
        os.path.normpath(os.path.join(here, "include", "config", "queries.yaml")),
        # Common image/container locations (COPY/MOUNT cases)
        "/usr/local/airflow/include/config/queries.yaml",
        "/opt/airflow/include/config/queries.yaml",
    ]

    for p in candidates:
        if os.path.exists(p):
            print(f"[news_raw_pipeline] queries.yaml -> {p}")
            return p

    # Fallback: read full YAML text from an Airflow Variable if provided
    inline_yaml = Variable.get("QUERIES_YAML", default_var=None)
    if inline_yaml:
        tmp = "/tmp/queries_from_variable.yaml"
        with open(tmp, "w") as f:
            f.write(inline_yaml)
        print("[news_raw_pipeline] queries.yaml loaded from Variable QUERIES_YAML")
        return tmp

    raise FileNotFoundError(
        "queries.yaml not found. Looked at:\n  - "
        + "\n  - ".join(candidates)
        + "\n(or set Airflow Variable 'QUERIES_YAML' with the file contents)"
    )


# ---------------------------------------------------------------------
# Loader for presets/defaults inside queries.yaml
# ---------------------------------------------------------------------
def _load_presets_from_yaml(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    defaults = data.get("defaults", {}) or {}
    presets = data.get("presets", []) or []

    # Allow overrides via Airflow Variables or env
    raw_days = Variable.get("RAW_DAYS", default_var=os.getenv("RAW_DAYS"))
    raw_max = Variable.get("RAW_MAX_RESULTS", default_var=os.getenv("RAW_MAX_RESULTS"))
    if raw_days is not None:
        defaults["days"] = int(raw_days)
    if raw_max is not None:
        defaults["max_results"] = int(raw_max)

    merged = []
    for p in presets:
        merged.append(
            {
                "topic": p["topic"],
                "query": p["query"],
                "days": int(p.get("days", defaults.get("days", 3))),
                "max_results": int(p.get("max_results", defaults.get("max_results", 20))),
            }
        )

    print(
        f"[news_raw_pipeline] loaded {len(merged)} presets, "
        f"defaults={{days:{defaults.get('days')}, max_results:{defaults.get('max_results')}}}"
    )
    return defaults, merged


# ---------------------------------------------------------------------
# Cloud Run invocation (RAW fetcher)
# ---------------------------------------------------------------------
def invoke_raw_fetch_news():
    # Use Variable → ENV → default URL (update default if you redeploy the service)
    url = (
        Variable.get("RAW_FETCH_NEWS_URL", default_var=None)
        or os.getenv("RAW_FETCH_NEWS_URL")
        or "https://raw-fetch-news-265141170939.us-central1.run.app"
    )
    print(f"[invoke_raw_fetch_news] Cloud Run URL: {url}")

    cfg_path = _queries_yaml_path()
    print(f"[invoke_raw_fetch_news] using queries.yaml at: {cfg_path}")
    defaults, presets = _load_presets_from_yaml(cfg_path)

    payload = {
        "defaults": {
            "days": int(defaults.get("days", 3)),
            "max_results": int(defaults.get("max_results", 20)),
        },
        "presets": presets,
    }

    # Increase timeout to tolerate slow external calls inside the service
    resp = requests.post(url, json=payload, timeout=300)
    print(f"[invoke_raw_fetch_news] HTTP {resp.status_code}")
    if resp.status_code not in (200, 204):
        raise RuntimeError(f"[raw] CF call failed: {resp.status_code} {resp.text}")


# ---------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------
with DAG(
    dag_id="news_raw_pipeline",
    description="Fetch news data into BigQuery RAW layer via Cloud Run, then trigger landing.",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="15 8 * * *",  # Daily at 08:15 UTC
    catchup=False,
    is_paused_upon_creation=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["news", "raw"],
) as dag:

    call_cf = PythonOperator(
        task_id="invoke_raw_fetch_news",
        python_callable=invoke_raw_fetch_news,
        execution_timeout=timedelta(minutes=6),
    )

    trigger_landing = TriggerDagRunOperator(
        task_id="trigger_news_landing_pipeline",
        trigger_dag_id="news_landing_pipeline",
        reset_dag_run=True,      # idempotent re-runs
        wait_for_completion=False,
    )

    call_cf >> trigger_landing