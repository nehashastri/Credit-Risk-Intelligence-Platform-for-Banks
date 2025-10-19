from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import requests
import os

def invoke_raw_fetch_news():
    url = Variable.get("RAW_FETCH_NEWS_URL", default_var=os.getenv("RAW_FETCH_NEWS_URL"))
    if not url:
        raise RuntimeError("RAW_FETCH_NEWS_URL not set (Airflow Variable or env)")
    resp = requests.post(url, json={}, timeout=60)
    if resp.status_code not in (200, 204):
        raise RuntimeError(f"CF call failed: {resp.status_code} {resp.text}")

with DAG(
    dag_id="news_raw_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="15 8 * * *",   
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["news","raw"],
) as dag:

    call_cf = PythonOperator(
        task_id="invoke_raw_fetch_news",
        python_callable=invoke_raw_fetch_news,
    )
