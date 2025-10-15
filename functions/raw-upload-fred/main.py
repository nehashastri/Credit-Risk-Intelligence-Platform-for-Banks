import functions_framework
from google.cloud import bigquery
import os
import json

PROJECT_ID = 'pipeline-882-team-project'
DATASET = 'raw'
BUCKET_NAME = 'group11-ba882-fall25-data'

@functions_framework.http
def task(request):
    """Load one FRED CSV from GCS into BigQuery."""
    request_args = request.args
    series_id = request_args.get("series_id")
    if not series_id:
        return {"error": "Missing series_id"}, 400

    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{DATASET}.{series_id.lower()}"
    uri = f"gs://{BUCKET_NAME}/fred/raw/{series_id}/*.csv"

    job_config = bigquery.LoadJobConfig(
        autodetect=True,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition="WRITE_APPEND",
    )

    load_job = client.load_table_from_uri(uri, table_id, job_config=job_config)
    load_job.result()

    table = client.get_table(table_id)
    return {"series_id": series_id, "rows": table.num_rows}, 200
