import functions_framework
import yfinance as yf
import pandas as pd
from google.cloud import storage, bigquery
from datetime import datetime, timedelta
import os

PROJECT_ID = "pipeline-882-team-project"
BUCKET_NAME = "group11-ba882-fall25-data"

@functions_framework.http
def raw_fetch_yfinance_append(request):
    """
    Fetches new daily close and volume data for the given ticker from Yahoo Finance.
    Determines the last date in BigQuery (raw.yfinance_table) and fetches data from that date onward.
    
    Request args:
      ?ticker=AAPL
    """
    request_args = request.args
    ticker = request_args.get("ticker")

    if not ticker:
        return {"error": "Missing ticker parameter"}, 400

    print(f"🔹 Starting yfinance append for ticker: {ticker}")

    try:
        # Initialize BigQuery client
        bq_client = bigquery.Client(project=PROJECT_ID)

        # Query to get the most recent date for this ticker
        query = f"""
            SELECT MAX(date) AS last_date
            FROM `pipeline-882-team-project.raw.yfinance_table`
            WHERE ticker = @ticker
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
        )

        query_job = bq_client.query(query, job_config=job_config)
        result = query_job.result()
        last_date_row = list(result)[0]
        last_date = last_date_row.last_date

        if last_date is None:
            print("No previous data found — fetching last 30 days.")
            start_date = datetime.utcnow().date() - timedelta(days=30)
        else:
            # Start from the next day after last_date
            start_date = last_date + timedelta(days=1)

        end_date = datetime.utcnow().date()
        print(f"Fetching yfinance data for {ticker} from {start_date} to {end_date}")

        if start_date >= end_date:
            return {
                "ticker": ticker,
                "status": "up_to_date",
                "message": f"No new data since {last_date}"
            }, 200

        # Fetch data
        df = yf.download(
            tickers=ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            progress=False
        )

        if df.empty:
            print("No new data returned by yfinance.")
            return {"ticker": ticker, "status": "no_data"}, 200

        # Clean and format
        df.reset_index(inplace=True)
        df.rename(columns={
            "Date": "date",
            "Close": "close_price",
            "Volume": "volume"
        }, inplace=True)
        df["date"] = df["date"].dt.date
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

        print(f"✅ Uploaded {len(df)} new rows to {blob_path}")

        return {
            "ticker": ticker,
            "rows_uploaded": len(df),
            "first_date": str(df['date'].min()),
            "last_date": str(df['date'].max()),
            "gcs_path": blob_path
        }, 200

    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        return {"error": str(e)}, 500
