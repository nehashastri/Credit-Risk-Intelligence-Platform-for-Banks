from airflow import DAG
from airflow.decorators import task
from datetime import datetime
import yaml
import os

CONFIG_PATH = "/opt/airflow/config/fred_series.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

SERIES_LIST = config["series"]
PROJECT_ID = config["project_id"]
BUCKET = config["bucket_name"]
DATASET = config["dataset_id"]


@task()
def fetch_fred(series_id: str, bucket_name: str):
    """Fetch one FRED series and upload to GCS."""
    import pandas as pd
    import requests
    from google.cloud import storage
    from datetime import datetime
    import os

    FRED_API_KEY = os.getenv("FRED_API_KEY")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json"}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["observations"]

    df = pd.DataFrame(data)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["ingest_timestamp"] = datetime.utcnow().isoformat()

    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    blob_name = f"fred/raw/{series_id}/{series_id}_{ts}.csv"
    bucket.blob(blob_name).upload_from_string(df.to_csv(index=False), content_type="text/csv")
    print(f"Uploaded {series_id} → gs://{bucket_name}/{blob_name}")
    return series_id


@task()
def load_to_bigquery(series_id: str, project_id: str, dataset_id: str, bucket_name: str):
    """Load one FRED series from GCS to BigQuery."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_id}.{series_id.lower()}"
    uri = f"gs://{bucket_name}/fred/raw/{series_id}/*.csv"

    job_config = bigquery.LoadJobConfig(
        autodetect=True,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )

    load_job = client.load_table_from_uri(uri, table_id, job_config=job_config)
    load_job.result()
    print(f"Loaded {load_job.output_rows} rows into {table_id}")


with DAG(
    dag_id="fred_ingest_dag",
    start_date=datetime(2025, 10, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["fred", "gcs", "bigquery"],
) as dag:

    fetched = fetch_fred.expand(
        series_id=SERIES_LIST,
        bucket_name=[BUCKET] * len(SERIES_LIST),
    )

    load_to_bigquery.expand(
        series_id=SERIES_LIST,
        project_id=[PROJECT_ID] * len(SERIES_LIST),
        dataset_id=[DATASET] * len(SERIES_LIST),
        bucket_name=[BUCKET] * len(SERIES_LIST),
    ).set_upstream(fetched)
