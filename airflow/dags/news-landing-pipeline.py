# dags/news-landing-pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.functions import CloudFunctionInvokeFunctionOperator

PROJECT_ID = "pipeline-882-team-project"
REGION = "us-central1"
FUNCTION_NAME = "landing-load-news"  # ← The Cloud Function deployed for the landing process

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="news_landing_pipeline",
    default_args=default_args,
    description="Load data from raw → landing for news and aggregate the last 7 days",
    schedule="30 8 * * *",  # Recommended: run 15 minutes after the raw pipeline (08:15 UTC)
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["news", "landing", "bigquery"],
) as dag:

    # Task to invoke the Cloud Function (landing-load-news)
    trigger_landing_load = CloudFunctionInvokeFunctionOperator(
        task_id="trigger_landing_load",
        project_id=PROJECT_ID,
        location=REGION,
        function_id=FUNCTION_NAME,
        input_data={},  # HTTP request body (empty JSON)
    )

    trigger_landing_load