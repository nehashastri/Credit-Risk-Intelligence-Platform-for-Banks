import functions_framework
import requests
import pandas as pd
from google.cloud import storage
from datetime import datetime
import os
import json

PROJECT_ID = 'pipeline-882-team-project'
BUCKET_NAME = 'group11-ba882-fall25-data'

@functions_framework.http
def task(request):
    """Fetch a FRED series and upload as CSV to GCS."""
    request_args = request.args
    series_id = request_args.get("series_id")
    if not series_id:
        return {"error": "Missing series_id"}, 400

    api_key = os.getenv("FRED_API_KEY")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json().get("observations", [])

    df = pd.DataFrame(data)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["ingest_timestamp"] = datetime.utcnow().isoformat()

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    blob_path = f"fred/raw/{series_id}/{series_id}_{ts}.csv"
    bucket.blob(blob_path).upload_from_string(df.to_csv(index=False), content_type="text/csv")

    return {"series_id": series_id, "gcs_path": blob_path}, 200
