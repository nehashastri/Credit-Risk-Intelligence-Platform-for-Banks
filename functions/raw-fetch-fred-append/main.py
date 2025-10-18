import functions_framework
import requests
import pandas as pd
from google.cloud import storage
from datetime import datetime, timedelta
from io import StringIO
import os

PROJECT_ID = "pipeline-882-team-project"
BUCKET_NAME = "group11-ba882-fall25-data"

@functions_framework.http
def task(request):
    """
    Fetch incremental FRED data and upload new records to GCS.
    Steps:
      1. Find latest CSV in GCS for given series_id.
      2. Determine last date present.
      3. Fetch only new data from FRED API.
      4. Upload incremental CSV to GCS.
    """
    request_args = request.args
    series_id = request_args.get("series_id")

    if not series_id:
        return {"error": "Missing series_id"}, 400

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return {"error": "Missing FRED_API_KEY environment variable"}, 500

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    prefix = f"fred/raw/{series_id}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    last_date = None

    if blobs:
        latest_blob = max(blobs, key=lambda b: b.time_created)
        try:
            content = latest_blob.download_as_text()
            df_prev = pd.read_csv(StringIO(content))
            if not df_prev.empty and "date" in df_prev.columns:
                last_date = df_prev["date"].max()
                print(f"Last date in previous CSV for {series_id}: {last_date}")
        except Exception as e:
            print(f"Could not read latest CSV: {e}")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json"
    }

    if last_date:
        try:
            overlap_date = (datetime.strptime(last_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
        except Exception:
            overlap_date = last_date
        params["observation_start"] = overlap_date
        print(f"Fetching data starting from {params['observation_start']}")
    else:
        print("No previous data found — fetching full series.")

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json().get("observations", [])
    if not data:
        return {"series_id": series_id, "message": "No data returned from FRED"}, 200

    df_new = pd.DataFrame(data)[["date", "value"]]
    df_new["value"] = pd.to_numeric(df_new["value"], errors="coerce")

    if last_date:
        df_new = df_new[df_new["date"] > last_date]

    if df_new.empty:
        print(f"No new rows to upload for {series_id}.")
        return {"series_id": series_id, "message": "No new data to append"}, 200

    df_new["ingest_timestamp"] = datetime.utcnow().isoformat()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    blob_path = f"fred/raw/{series_id}/{series_id}_{ts}.csv"
    bucket.blob(blob_path).upload_from_string(df_new.to_csv(index=False), content_type="text/csv")

    print(f"Uploaded incremental file: {blob_path}")
    return {
        "series_id": series_id,
        "rows_uploaded": len(df_new),
        "first_date": df_new['date'].min(),
        "last_date": df_new['date'].max(),
        "gcs_path": blob_path
    }, 200
