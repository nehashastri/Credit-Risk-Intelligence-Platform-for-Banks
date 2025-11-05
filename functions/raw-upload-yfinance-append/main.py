import functions_framework
import pandas as pd
from google.cloud import storage, bigquery
from datetime import datetime
from io import StringIO

PROJECT_ID = "pipeline-882-team-project"
BUCKET_NAME = "group11-ba882-fall25-data"
DATASET_ID = "raw"
TABLE_ID = "yfinance_table"

@functions_framework.http
def raw_upload_yfinance_append(request):
    """
    Reads the most recent YFinance CSV for the given ticker from GCS
    and appends it into BigQuery (raw.sector_equity_features_2).
    """

    request_args = request.args
    ticker = request_args.get("ticker")

    if not ticker:
        return {"error": "Missing ticker parameter"}, 400

    print(f"🚀 Starting upload for {ticker}")

    try:
        # --- Initialize clients ---
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        bq_client = bigquery.Client(project=PROJECT_ID)

        # --- Find latest uploaded CSV in GCS ---
        prefix = f"yfinance/raw/{ticker}/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            print(f"⚠️ No files found under {prefix}")
            return {"ticker": ticker, "message": "No CSVs found in GCS"}, 200

        latest_blob = max(blobs, key=lambda b: b.time_created)
        blob_path = latest_blob.name
        print(f"📦 Latest blob found: {blob_path}")

        # --- Read CSV into DataFrame ---
        content = latest_blob.download_as_text()
        df = pd.read_csv(StringIO(content))

        if df.empty:
            print(f"⚠️ Empty CSV for {ticker}")
            return {"ticker": ticker, "message": "Empty CSV"}, 200

        # --- Clean: drop rows where 'date' == ticker or non-date garbage ---
        invalid_rows = df[df['date'].astype(str).str.upper() == ticker.upper()]
        if not invalid_rows.empty:
            print(f"🧹 Removing {len(invalid_rows)} invalid ticker rows for {ticker}")
            df = df[df['date'].astype(str).str.upper() != ticker.upper()]

        # Optionally drop rows that don't parse as valid dates
        df = df[pd.to_datetime(df['date'], errors='coerce').notna()]

        if df.empty:
            print(f"⚠️ No valid rows left after cleanup for {ticker}")
            return {"ticker": ticker, "message": "No valid rows after cleanup"}, 200

        # --- Ensure correct schema alignment ---
        expected_cols = [
            "date",
            "ticker",
            "close_price",
            "volume",
            "ingest_timestamp"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        df = df[expected_cols]

        # --- Load into BigQuery ---
        table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            source_format=bigquery.SourceFormat.CSV,
            schema=[
                bigquery.SchemaField("date", "DATE"),
                bigquery.SchemaField("ticker", "STRING"),
                bigquery.SchemaField("close_price", "FLOAT"),
                bigquery.SchemaField("volume", "FLOAT"),
                bigquery.SchemaField("ingest_timestamp", "TIMESTAMP")
            ],
        )

        load_job = bq_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        load_job.result()  # Wait for job completion

        print(f"✅ Uploaded {len(df)} cleaned rows for {ticker} → {table_ref}")

        return {
            "ticker": ticker,
            "rows_uploaded": len(df),
            "bq_table": table_ref,
            "gcs_source": blob_path,
            "status": "success",
        }, 200

    except Exception as e:
        print(f"❌ Error uploading {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}, 500
