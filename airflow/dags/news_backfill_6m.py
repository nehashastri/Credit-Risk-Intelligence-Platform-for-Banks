# dags/news-backfill-6m.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import os, time, requests
from google.cloud import bigquery

CF_URL = os.getenv("RAW_FETCH_NEWS_URL", "https://raw-fetch-news-r25wwaz52q-uc.a.run.app")

BACKFILL_DAYS = int(os.getenv("NEWS_BACKFILL_DAYS", 185))
MAX_PER_TOPIC = int(os.getenv("NEWS_BACKFILL_MAX", 30))

PROJECT_ID   = os.getenv("GCP_PROJECT_ID", "pipeline-882-team-project")
RAW_DATASET  = os.getenv("RAW_DATASET", "raw")
RAW_TABLE    = os.getenv("RAW_TABLE", "raw_news_articles")
RAW_FQN      = f"`{PROJECT_ID}.{RAW_DATASET}.{RAW_TABLE}`"

PRESETS = [
    {"topic":"fed_policy",  "query":"Federal Reserve FOMC rate decision OR Fed policy"},
    {"topic":"cpi",         "query":"US CPI inflation BLS report"},
    {"topic":"labor",       "query":"US unemployment labor market JOLTS NFP"},
    {"topic":"markets",     "query":"US stock market treasury yields credit spreads"},
    {"topic":"energy",      "query":"oil prices gasoline diesel energy market"},
    {"topic":"real_estate", "query":"US housing market mortgage delinquency"},
]

def backfill_6m_chunked():
    for p in PRESETS:
        payload = {
            "defaults": {"days": BACKFILL_DAYS, "max_results": MAX_PER_TOPIC},
            "presets":  [{**p, "days": BACKFILL_DAYS, "max_results": MAX_PER_TOPIC}],
        }
        r = requests.post(CF_URL, json=payload, timeout=540)
        print(f"[backfill_6m] topic={p['topic']} status={r.status_code} body={r.text[:300]}")
        if r.status_code not in (200, 204):
            raise RuntimeError(f"backfill failed for {p['topic']}: {r.status_code} {r.text}")
        time.sleep(1)

DELETE_THIS_MONTH_SQL = f"""
DECLARE _month_start DATE DEFAULT DATE_TRUNC(CURRENT_DATE(), MONTH);

DELETE FROM {RAW_FQN}
WHERE ingest_date = CURRENT_DATE()
  AND published_at IS NOT NULL
  AND DATE(published_at) >= _month_start;
"""

def prune_current_month_rows():
    client = bigquery.Client(project=PROJECT_ID)
    job = client.query(DELETE_THIS_MONTH_SQL)
    job.result()
    print("[prune] deleted current-month rows ingested today.")

default_args = {"retries": 0, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="news_backfill_6m",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["news","raw","backfill"],
) as dag:
    run_backfill = PythonOperator(
        task_id="invoke_raw_backfill_6m",
        python_callable=backfill_6m_chunked,
    )

    prune_this_month = PythonOperator(
        task_id="prune_current_month_rows",
        python_callable=prune_current_month_rows,
    )

    run_backfill >> prune_this_month