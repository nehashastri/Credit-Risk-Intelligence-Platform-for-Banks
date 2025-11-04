import functions_framework
import yfinance as yf
import pandas as pd
from google.cloud import storage
from datetime import datetime, timedelta
import os
from io import StringIO

PROJECT_ID = "pipeline-882-team-project"
BUCKET_NAME = "group11-ba882-fall25-data"

@functions_framework.http
def raw_fetch_yfinance_append(request):
    """
    Fetches daily close and volume data for the given ticker from Yahoo Finance.
    Appends data for yesterday only and uploads to GCS.
    
    Request args:
      ?ticker=AAPL
    """
    request_args = request.args
    ticker = request_args.get("ticker")

    if not ticker:
        return {"error": "Missing ticker parameter"}, 400

    # Determine yesterday's date (UTC)
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=1)

    print(f"Fetching yfinance data for {ticker} from {start_date} to {end_date}")

    try:
        # Fetch 1 day of data
        df = yf.download(
            tickers=ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            progress=False
        )

        if df.empty:
            return {"ticker": ticker, "status": "no_data", "message": "No data returned for yesterday"}, 200


        df.reset_index(inplace=True)
        df.rename(columns={
            "Date": "date",
            "Close": "close_price",
            "Volume": "volume"
        }, inplace=True)

        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df["ticker"] = ticker
        df["ingest_timestamp"] = datetime.utcnow().isoformat()

        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        blob_path = f"yfinance/raw/{ticker}/{ticker}_{ts}.csv"

        bucket.blob(blob_path).upload_from_string(
            df.to_csv(index=False),
            content_type="text/csv"
        )

        print(f"Uploaded {len(df)} rows to {blob_path}")

        return {
            "ticker": ticker,
            "rows_uploaded": len(df),
            "first_date": df["date"].min(),
            "last_date": df["date"].max(),
            "gcs_path": blob_path
        }, 200

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return {"error": str(e)}, 500
