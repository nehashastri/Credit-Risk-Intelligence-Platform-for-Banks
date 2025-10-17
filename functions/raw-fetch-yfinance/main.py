import functions_framework
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from google.cloud import bigquery
import json

# Configuration
PROJECT_ID = "pipeline-882-team-project"
RAW_DATASET = "raw"
RAW_TABLE = "sector_equity_features"

# Tickers / sector ETFs
TICKERS = ["SPY", "XLY", "XLP", "XLF", "XLE", "XLK", "XLI", "XLRE", "XLB", "XLU", "XLV"]

@functions_framework.http
def task(request):
    try:
        print("Step 1: Reading parameters")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        if not end_date:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")
        if not start_date:
            dt_end = datetime.fromisoformat(end_date)
            dt_start = dt_end - timedelta(days=30)
            start_date = dt_start.strftime("%Y-%m-%d")

        print(f"Fetching tickers {TICKERS} from {start_date} to {end_date}")
        df = yf.download(
            TICKERS,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        # Flatten the multi-index df
        records = []
        for ticker in TICKERS:
            try:
                dft = df[ticker]
            except Exception:
                dft = df
            for date, row in dft.iterrows():
                records.append({
                    "date": date.date().isoformat(),
                    "ticker": ticker,
                    "open_price": float(row["Open"]),
                    "high_price": float(row["High"]),
                    "low_price": float(row["Low"]),
                    "close_price": float(row["Close"]),
                    "volume": int(row["Volume"]),
                })

        price_df = pd.DataFrame.from_records(records)
        price_df = price_df.sort_values(["ticker", "date"]).reset_index(drop=True)

        def compute_drawdown(series: pd.Series, window_size: int):
            # drawdown = (current / rolling_max) - 1
            return (series / series.rolling(window_size).max()) - 1

        # Compute features
        price_df["return_1d"] = price_df.groupby("ticker")["close_price"].pct_change(1)
        price_df["return_1w"] = price_df.groupby("ticker")["close_price"].pct_change(7)
        price_df["return_1m"] = price_df.groupby("ticker")["close_price"].pct_change(30)
        price_df["volatility_1w"] = price_df.groupby("ticker")["return_1d"].rolling(7).std().reset_index(level=0, drop=True)
        price_df["volatility_1m"] = price_df.groupby("ticker")["return_1d"].rolling(30).std().reset_index(level=0, drop=True)
        price_df["momentum_4w"] = price_df.groupby("ticker")["close_price"].pct_change(28)
        price_df["drawdown_max_4w"] = price_df.groupby("ticker")["close_price"].transform(lambda ser: compute_drawdown(ser, 28))

        price_df["ingest_timestamp"] = datetime.utcnow().isoformat()

        # BigQuery client
        client = bigquery.Client(project=PROJECT_ID)
        dataset_ref = client.dataset(RAW_DATASET)
        table_ref = f"{PROJECT_ID}.{RAW_DATASET}.{RAW_TABLE}"

        # Ensure dataset exists
        try:
            client.get_dataset(dataset_ref)
        except Exception:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            client.create_dataset(dataset)
            print(f"Created dataset {RAW_DATASET}")

        # Ensure table exists
        try:
            client.get_table(table_ref)
            print(f"Table {RAW_TABLE} exists")
        except Exception:
            # Define schema dynamically based on DataFrame
            schema = [
                bigquery.SchemaField(col, "STRING") if str(price_df[col].dtype) == "object"
                else bigquery.SchemaField(col, "FLOAT") if "price" in col or "return" in col or "volatility" in col or "drawdown" in col
                else bigquery.SchemaField(col, "INTEGER") if "volume" in col
                else bigquery.SchemaField(col, "TIMESTAMP") if col == "ingest_timestamp"
                else bigquery.SchemaField(col, "DATE") if col == "date"
                else bigquery.SchemaField(col, "STRING")
                for col in price_df.columns
            ]
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table)
            print(f"Created table {RAW_TABLE}")

        # Load into BigQuery
        job = client.load_table_from_dataframe(price_df, table_ref, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"))
        job.result()
        print(f"Loaded {job.output_rows} rows into {table_ref}")

        return {"rows_loaded": job.output_rows}, 200

    except Exception as e:
        print("Error in raw-fetch-yfinance:", str(e))
        return {"error": str(e)}, 500
