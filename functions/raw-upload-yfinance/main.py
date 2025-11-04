import functions_framework
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
import os
import json

# --- Environment Variables ---
PROJECT_ID = os.getenv("PROJECT_ID", "pipeline-882-team-project")
DATASET_ID = os.getenv("DATASET_ID", "raw")
TABLE_NAME = "sector_equity_features_1"
BUCKET_NAME = os.getenv("BUCKET_NAME", "group11-ba882-fall25-data")

@functions_framework.http
def raw_upload_yfinance_append(request):
    """
    Triggered via HTTP by Airflow DAG.
    Appends the latest YFinance CSV (already uploaded to GCS)
    into BigQuery table 'raw.sector_equity_features_1'.
    """

    request_json = request.get_json(silent=True)
    request_args = request.args

    # --- Get ticker ---
    ticker = None
    if request_json and "ticker" in request_json:
        ticker = request_json["ticker"]
    elif request_args and "ticker" in request_args:
        ticker = request_args["ticker"]

    if not ticker:
        return ("Missing 'ticker' parameter", 400)

    # --- Construct GCS URI ---
    file_name = f"yfinance/raw/{ticker}/{ticker}_latest.csv"
    uri = f"gs://{BUCKET_NAME}/{file_name}"
    print(f"🚀 Starting load for {ticker} from {uri}")

    client = bigquery.Client(project=PROJECT_ID)
    dataset_ref = client.dataset(DATASET_ID)
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"

    # --- Ensure dataset exists ---
    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"✅ Created dataset: {DATASET_ID}")

    # --- Ensure table exists ---
    try:
        client.get_table(table_id)
        print(f"📄 Table {table_id} already exists.")
    except NotFound:
        schema = [
            bigquery.SchemaField("date", "DATE"),
            bigquery.SchemaField("ticker", "STRING"),
            bigquery.SchemaField("open_price", "FLOAT"),
            bigquery.SchemaField("high_price", "FLOAT"),
            bigquery.SchemaField("low_price", "FLOAT"),
            bigquery.SchemaField("close_price", "FLOAT"),
            bigquery.SchemaField("volume", "FLOAT"),
            bigquery.SchemaField("return_1d", "FLOAT"),
            bigquery.SchemaField("return_1w", "FLOAT"),
            bigquery.SchemaField("return_1m", "FLOAT"),
            bigquery.SchemaField("volatility_1w", "FLOAT"),
            bigquery.SchemaField("volatility_1m", "FLOAT"),
            bigquery.SchemaField("momentum_4w", "FLOAT"),
            bigquery.SchemaField("drawdown_max_4w", "FLOAT"),
            bigquery.SchemaField("ingest_timestamp", "TIMESTAMP"),
        ]
        client.create_table(bigquery.Table(table_id, schema=schema))
        print(f"✅ Created table: {table_id}")

    # --- Load CSV into BigQuery ---
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=False,
        write_disposition="WRITE_APPEND",
        field_delimiter=",",
        schema_update_options=[],
    )

    print(f"📥 Loading data from {uri} into {table_id}...")
    load_job = client.load_table_from_uri(uri, table_id, job_config=job_config)
    load_job.result()

    print(f"✅ Successfully appended data for {ticker}.")
    return (json.dumps({"status": "success", "ticker": ticker}), 200)
