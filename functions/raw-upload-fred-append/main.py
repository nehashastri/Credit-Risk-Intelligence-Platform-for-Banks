import functions_framework
from google.cloud import bigquery, storage
from google.api_core.exceptions import NotFound
from datetime import datetime
import os, json

PROJECT_ID = os.getenv("PROJECT_ID", "pipeline-882-team-project")
DATASET_ID = os.getenv("DATASET_ID", "raw")
BUCKET_NAME = os.getenv("BUCKET_NAME", "group11-ba882-fall25-data")

@functions_framework.http
def raw_upload_fred_append(request):
    """
    Triggered via HTTP by Airflow DAG.
    Appends the latest FRED CSV (already uploaded to GCS) into BigQuery.
    """
    request_json = request.get_json(silent=True)
    request_args = request.args

    series_id = (
        (request_json or {}).get("series_id")
        or (request_args or {}).get("series_id")
    )
    if not series_id:
        return ("Missing series_id parameter", 400)

    # 🔹 Find latest CSV file in GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    prefix = f"fred/raw/{series_id}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        return (f"No CSV files found in GCS for {series_id}", 404)

    latest_blob = max(blobs, key=lambda b: b.time_created)
    uri = f"gs://{BUCKET_NAME}/{latest_blob.name}"
    print(f"📦 Found latest CSV for {series_id}: {uri}")

    client = bigquery.Client(project=PROJECT_ID)
    dataset_ref = client.dataset(DATASET_ID)
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{series_id.lower()}"

    # Ensure dataset exists
    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"✅ Created dataset: {DATASET_ID}")

    # Ensure table exists
    try:
        client.get_table(table_id)
        print(f"📄 Table {table_id} already exists.")
    except NotFound:
        schema = [
            bigquery.SchemaField("date", "DATE"),
            bigquery.SchemaField("value", "FLOAT"),
            bigquery.SchemaField("ingest_timestamp", "TIMESTAMP"),
        ]
        client.create_table(bigquery.Table(table_id, schema=schema))
        print(f"✅ Created table: {table_id}")

    # Load CSV from GCS → BigQuery
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=False,
        write_disposition="WRITE_APPEND",
    )

    print(f"📥 Loading data from {uri} into {table_id}...")
    load_job = client.load_table_from_uri(uri, table_id, job_config=job_config)
    load_job.result()
    print(f"✅ Appended {load_job.output_rows} rows for {series_id}.")

    return (
        json.dumps({
            "status": "success",
            "series_id": series_id,
            "rows_appended": load_job.output_rows,
            "gcs_file": latest_blob.name,
            "bq_table": table_id,
        }),
        200,
    )
