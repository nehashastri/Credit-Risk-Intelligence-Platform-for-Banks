# dags/news-landing-pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import requests
import os

PROJECT_ID = "pipeline-882-team-project"

def invoke_landing_load_news():
    url = (Variable.get("LANDING_LOAD_NEWS_URL", default_var=None)
           or os.getenv("LANDING_LOAD_NEWS_URL")
           or "https://landing-load-news-r25wwaz52q-uc.a.run.app")
    payload = {}
    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code not in (200, 204):
        raise RuntimeError(f"[landing] CF call failed: {resp.status_code} {resp.text}")

with DAG(
    dag_id="news_landing_pipeline",
    description="Load data from raw → landing for news and aggregate the last 7 days",
    schedule="30 8 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["news", "landing", "bigquery"],
) as dag:

    trigger_landing_load = PythonOperator(
        task_id="invoke_landing_load_news",
        python_callable=invoke_landing_load_news,
    )