import os
import io
import pandas as pd
import requests
from datetime import datetime
from google.cloud import storage
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/fred_series.yaml")
FRED_API_KEY = os.getenv("FRED_API_KEY")

def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Fetch a single FRED series as a DataFrame with ingest_timestamp."""
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()["observations"]
    df = pd.DataFrame(data)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["ingest_timestamp"] = datetime.utcnow().isoformat()
    return df

def upload_to_gcs(bucket_name, blob_name, df):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    csv_data = df.to_csv(index=False)
    blob.upload_from_string(csv_data, content_type="text/csv")

def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    bucket = config["bucket_name"]
    series_list = config["series"]

    for s in series_list:
        print(f"Fetching {s}...")
        df = fetch_fred_series(s)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        blob_name = f"fred/raw/{s}/{s}_{timestamp}.csv"
        upload_to_gcs(bucket, blob_name, df)
        print(f"Uploaded {s} → gs://{bucket}/{blob_name}")

if __name__ == "__main__":
    main()
