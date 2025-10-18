import functions_framework
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
import os
import json

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

    # Get series_id (from ?series_id=XYZ)
    series_id = None
    if request_json and "series_id" in request_json:
        series_id = request_json["series_id"]
    elif request_args and "series_id" in request_args:
        series_id = request_args["series_id"]

    if not series_id:
        return ("Missing series_id parameter", 400)

    file_name = f"fred/raw/{series_id}/{series_id}_latest.csv"
    uri = f"gs://{BUCKET_NAME}/{file_name}"

    print(f"🚀 Starting load for {series_id} from {uri}")

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

    # Load CSV from GCS → BQ append
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=False,
        write_disposition="WRITE_APPEND",
    )

    print(f"📥 Loading data from {uri} into {table_id}...")
    load_job = client.load_table_from_uri(uri, table_id, job_config=job_config)
    load_job.result()

    print(f"✅ Successfully appended {series_id} data.")
    return (json.dumps({"status": "success", "series_id": series_id}), 200)
