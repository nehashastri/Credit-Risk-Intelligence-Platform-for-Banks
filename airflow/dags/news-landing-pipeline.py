from airflow import DAG
from datetime import datetime, timedelta
from airflow.providers.standard.operators.python import PythonOperator
from airflow.models import Variable
import os, requests

def call_landing():
    url = Variable.get("LANDING_LOAD_NEWS_URL", default_var=os.getenv("LANDING_LOAD_NEWS_URL", "")).strip()
    if not url:
        raise RuntimeError("LANDING_LOAD_NEWS_URL not set (Airflow Variable or env)")
    r = requests.post(url, json={}, timeout=180)
    print(f"[landing] status={r.status_code}, body={r.text[:300]}")
    if r.status_code not in (200, 204):
        raise RuntimeError(f"landing-load-news failed: {r.status_code} {r.text}")

with DAG(
    dag_id="news_landing_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="30 8 * * *",             
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["news","landing"],
) as dag:

    landing_load = PythonOperator(
        task_id="invoke_landing_load_news",
        python_callable=call_landing,
    )
