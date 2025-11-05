from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import requests
import os
import yaml

# --- YAML loader: safely locate include/queries.yaml ---
def _queries_yaml_path() -> str:
    # Compute relative path based on this DAG file location
    here = os.path.dirname(os.path.abspath(__file__))
    # airflow/dags/ -> airflow/include/queries.yaml
    return os.path.normpath(os.path.join(here, "..", "include", "queries.yaml"))

def _load_presets_from_yaml(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    defaults = data.get("defaults", {}) or {}
    presets  = data.get("presets", []) or []

    # Allow Airflow Variables or environment variables to override defaults
    # (they take precedence over values in YAML)
    raw_days = Variable.get("RAW_DAYS", default_var=os.getenv("RAW_DAYS"))
    raw_max  = Variable.get("RAW_MAX_RESULTS", default_var=os.getenv("RAW_MAX_RESULTS"))
    if raw_days is not None:
        defaults["days"] = int(raw_days)
    if raw_max is not None:
        defaults["max_results"] = int(raw_max)

    # Inject defaults into each preset (use preset-specific values if provided)
    merged = []
    for p in presets:
        merged.append({
            "topic": p["topic"],
            "query": p["query"],
            "days": int(p.get("days", defaults.get("days", 3))),
            "max_results": int(p.get("max_results", defaults.get("max_results", 20))),
        })
    return defaults, merged

def invoke_raw_fetch_news():
    # Get Cloud Function URL from Airflow Variable → ENV → fallback default
    url = (Variable.get("RAW_FETCH_NEWS_URL", default_var=None)
           or os.getenv("RAW_FETCH_NEWS_URL")
           or "https://raw-fetch-news-r25wwaz52q-uc.a.run.app")  # Temporary fallback URL

    print(f"[invoke_raw_fetch_news] Using Cloud Function URL: {url}")

    # Load YAML → build request payload
    defaults, presets = _load_presets_from_yaml(_queries_yaml_path())

    payload = {
        "defaults": {
            "days": int(defaults.get("days", 3)),
            "max_results": int(defaults.get("max_results", 20)),
        },
        "presets": presets,
    }

    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code not in (200, 204):
        raise RuntimeError(f"CF call failed: {resp.status_code} {resp.text}")

with DAG(
    dag_id="news_raw_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="15 8 * * *",    # Run daily at 08:15 UTC
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["news", "raw"],
) as dag:

    call_cf = PythonOperator(
        task_id="invoke_raw_fetch_news",
        python_callable=invoke_raw_fetch_news,
    )