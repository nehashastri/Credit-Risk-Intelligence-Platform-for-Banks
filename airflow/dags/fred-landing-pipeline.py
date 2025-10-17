from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.functions import CloudFunctionInvokeFunctionOperator

# DAG settings
PROJECT_ID = "pipeline-882-team-project"
REGION = "us-central1"
FUNCTION_NAME = "landing-load-fred"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fred_landing_pipeline",
    default_args=default_args,
    description="Extracts from raw schema and loads macro indicators into landing schema",
    schedule="@daily",  # or set to None for manual runs
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["fred", "landing", "bigquery"],
) as dag:

    # This task calls your deployed Cloud Function
    trigger_landing_load = CloudFunctionInvokeFunctionOperator(
        task_id="trigger_landing_load",
        project_id=PROJECT_ID,
        location=REGION,
        input_data={},  # request body (optional)
        function_id=FUNCTION_NAME,
    )

    trigger_landing_load
