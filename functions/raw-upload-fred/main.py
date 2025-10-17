import functions_framework
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
import os
import json

PROJECT_ID = "pipeline-882-team-project"
DATASET_ID = "raw"
BUCKET_NAME = "group11-ba882-fall25-data"

@functions_framework.http
def task(request):
    """Load one FRED CSV from GCS into BigQuery with automatic table creation."""
    request_args = request.args
    series_id = request_args.get("series_id")
    if not series_id:
        return {"error": "Missing series_id"}, 400

    client = bigquery.Client(project=PROJECT_ID)
    dataset_ref = client.dataset(DATASET_ID)
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{series_id.lower()}"
    uri = f"gs://{BUCKET_NAME}/fred/raw/{series_id}/*.csv"

    # 1️⃣ Ensure the dataset exists
    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"Created dataset: {DATASET_ID}")

    # 2️⃣ Ensure the table exists (CREATE TABLE IF NOT EXISTS equivalent)
    try:
        client.get_table(table_id)
        print(f"Table {table_id} exists.")
    except NotFound:
        print(f"Creating new table {table_id}...")
        # Define basic schema matching the FRED CSVs
        schema = [
            bigquery.SchemaField("date", "DATE"),
            bigquery.SchemaField("value", "FLOAT"),
            bigquery.SchemaField("ingest_timestamp", "TIMESTAMP"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)
        print(f"Created table: {table_id}")

    # 3️⃣ Load data from GCS → BigQuery
    job_config = bigquery.LoadJobConfig(
        autodetect=False,  # we know the schema
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition="WRITE_APPEND",
    )

    load_job = client.load_table_from_uri(uri, table_id, job_config=job_config)
    load_job.result()

    # 4️⃣ Return summary
    table = client.get_table(table_id)
    return {
        "series_id": series_id,
        "rows_after_load": table.num_rows,
        "table_id": table_id
    }, 200
