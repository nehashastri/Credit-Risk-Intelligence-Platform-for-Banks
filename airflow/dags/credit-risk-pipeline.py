# dags/credit-risk-pipeline.py
from airflow.decorators import dag, task
from datetime import datetime, timedelta
from airflow.operators.python import get_current_context
from airflow.exceptions import AirflowSkipException
from google.api_core.exceptions import TooManyRequests
import requests, yaml, json, time
from jinja2 import Template
import uuid
import utils 

# --------------------------------------------------
# Utility: Safe Cloud Function invoker
# --------------------------------------------------
def invoke_function(url, params={}, method="GET"):
    """Unified Cloud Function invoker with error handling."""
    try:
        if method.upper() == "POST":
            resp = requests.post(url, json=params)
        else:
            resp = requests.get(url, params=params)

        # Handle known skip conditions
        if resp.status_code == 204 or "no new" in resp.text.lower():
            raise AirflowSkipException("No new data.")
        elif resp.status_code == 500:
            raise AirflowSkipException("Server 500 error.")
        elif resp.status_code >= 400:
            print(f"❌ Request failed ({resp.status_code}): {resp.text}")
            resp.raise_for_status()

        try:
            return resp.json()
        except json.JSONDecodeError:
            return {"text": resp.text}

    except AirflowSkipException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error while calling {url}: {e}")
        raise

# --------------------------------------------------
# DAG Definition
# --------------------------------------------------
@dag(
    schedule="@daily",
    start_date=datetime(2025, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["credit-risk", "fred", "yfinance", "ml-dataset"],
)
def credit_risk_pipeline():
    """
    Combined Credit Risk Pipeline:
    1️⃣ FRED incremental pipeline (fetch → upload → landing)
    2️⃣ YFinance incremental pipeline (fetch → upload → landing)
    3️⃣ Invoke create_ml_dataset Cloud Function
    """

    # --------------------------------------------------
    # Config files
    # --------------------------------------------------
    FRED_CONFIG = "/usr/local/airflow/include/config/fred_series.yaml"
    YFIN_CONFIG = "/usr/local/airflow/include/config/yfinance_tickers.yaml"
    SQL_DIR = "/usr/local/airflow/include/sql" 

    with open(FRED_CONFIG, "r") as f:
        fred_series = yaml.safe_load(f)["series"]

    with open(YFIN_CONFIG, "r") as f:
        yfinance_tickers = yaml.safe_load(f)["tickers"]

    # --------------------------------------------------
    # FRED Extract & Load Tasks
    # --------------------------------------------------
    @task
    def extract_fred(series_id: str) -> dict:
        """Fetch new FRED data for a given series."""
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw-fetch-fred-append"
        ctx = get_current_context()
        payload = {"series_id": series_id, "run_id": ctx["dag_run"].run_id}
        try:
            resp = invoke_function(url, params=payload)
            return {"series_id": series_id, "new_data": True, "resp": resp}
        except AirflowSkipException:
            print(f"⏩ No new FRED data for {series_id}")
            raise
        return resp

    @task
    def load_fred_to_bq(payload: dict):
        """Append new FRED data to BigQuery."""
        series_id = payload["series_id"]
        if not payload.get("new_data"):
            raise AirflowSkipException(f"No new FRED data for {series_id}")
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw_upload_fred_append"
        resp = invoke_function(url, params={"series_id": series_id})
        return {"series_id": series_id, "loaded": True, "resp": resp}

    @task(trigger_rule="all_done")
    def load_fred_landing(results: list):
        """Load FRED landing only if any series updated."""
        if not any(r.get("loaded") for r in results if isinstance(r, dict)):
            raise AirflowSkipException("No FRED updates → skip landing.")
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/landing-load-fred"
        print("🚀 Running FRED landing transformation...")
        return invoke_function(url)

    # --------------------------------------------------
    # YFinance Extract & Load Tasks (Updated)
    # --------------------------------------------------
    @task
    def extract_yfinance(ticker: str) -> dict:
        """Fetch the latest YFinance data for a ticker and upload it to GCS."""
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw_fetch_yfinance_append"
        ctx = get_current_context()
        payload = {"ticker": ticker, "run_id": ctx["dag_run"].run_id}

        print(f"🚀 Fetching data for {ticker} ...")
        resp = invoke_function(url, params=payload)
        status = resp.get("status", "").lower()

        if status in ["no_data", "skipped", "up_to_date"] or "no new" in str(resp).lower():
            msg = f"⏩ No new data for {ticker} — skipping further steps (status={status})"
            print(msg)
            raise AirflowSkipException(msg)

        return resp

    @task(retries=2, retry_delay=timedelta(seconds=20))
    def load_yfinance_to_bq(payload: dict) -> dict:
        """Append uploaded YFinance CSV into BigQuery with retry on rate limits."""
        ticker = payload.get("ticker")
        status = payload.get("status", "").lower()

        if not ticker or status in ["no_data", "skipped", "up_to_date"]:
            raise AirflowSkipException(f"No new data for {ticker}")

        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/raw_upload_yfinance_append"

        # Retry loop for rateLimitExceeded (429)
        for attempt in range(1, 6):
            try:
                print(f"⬆️ [{ticker}] Attempt {attempt}: Uploading to BigQuery...")
                resp = invoke_function(url, params={"ticker": ticker})
                print(f"✅ [{ticker}] Upload success: {resp}")
                return {"ticker": ticker, "status": "success", "resp": resp}

            except requests.exceptions.HTTPError as e:
                if "429" in str(e) or "rateLimitExceeded" in str(e):
                    wait_time = 20 * attempt
                    print(f"⚠️ [{ticker}] Rate limit hit — retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ [{ticker}] Upload failed: {e}")
                    raise

        raise AirflowSkipException(f"[{ticker}] Failed after multiple retries due to rate limit.")

    @task(trigger_rule="all_done")
    def load_yfinance_landing(results: list):
        """Run landing transformation if any ticker had successful uploads."""
        if not any(r.get("status") == "success" for r in results if isinstance(r, dict)):
            raise AirflowSkipException("No YFinance updates → skip landing.")
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/landing_load_yfinance_append"
        print("🚀 Triggering YFinance landing transformation...")
        return invoke_function(url)

    # --------------------------------------------------
    # Final Step: Trigger ML Dataset Function
    # --------------------------------------------------
    @task(trigger_rule="all_done")
    def create_ml_dataset():
        """Trigger Cloud Function to build ML dataset."""
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/create_ml_dataset"
        print("🚀 Triggering ML dataset creation Cloud Function...")
        resp = invoke_function(url)
        print(f"✅ ML dataset creation response: {resp}")
        return resp

    # -----------------------------------------------------------
    # TASK 1: Register the model in mlops.model
    # -----------------------------------------------------------
    @task
    def register_model():
        """Register the credit delinquency model in mlops.model."""
        model_vals = {
            "model_id": "credit_delinquency_model",
            "model_name": "Credit Delinquency Rate Predictor",
        }

        s = utils.read_sql(f"{SQL_DIR}/mlops-model-registry.sql")
        template = Template(s)
        sql = template.render(**model_vals)

        print("🧾 Executing model registration SQL:")
        print(sql)
        utils.run_execute(sql)
        print("✅ Model registered successfully.")

    # -----------------------------------------------------------
    # TASK 2: Register the dataset snapshot in mlops.dataset
    # -----------------------------------------------------------
    @task
    def register_dataset():
        """Register ML dataset metadata in BigQuery (mlops.dataset)."""

        dataset_metadata = {
            "dataset_id": f"ds_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}",
            "data_version": datetime.now().strftime("%Y-%m-%d"),
            "row_count": utils.run_fetchone("""
                SELECT COUNT(*) 
                FROM `pipeline-882-team-project.gold.fact_all_indicators_weekly`
            """)[0],
            "feature_count": utils.run_fetchone("""
                SELECT COUNT(*) 
                FROM UNNEST(REGEXP_EXTRACT_ALL(
                    TO_JSON_STRING((SELECT AS STRUCT * 
                                    FROM `pipeline-882-team-project.gold.fact_all_indicators_weekly` 
                                    LIMIT 1)),
                    r'"[^"]*":'
                ))
            """)[0],
            "model_id": "credit_delinquency_model",
        }

        s = utils.read_sql(f"{SQL_DIR}/mlops-dataset-registry.sql")
        template = Template(s)
        sql = template.render(**dataset_metadata)

        print("🧾 Executing dataset registration SQL:")
        print(sql)
        utils.run_execute(sql)
        print("✅ Dataset registered successfully.")

    # -----------------------------------------------------------
    # TASK 3: Trigger ML Dataset creation Cloud Function
    # -----------------------------------------------------------
    @task(trigger_rule="all_done")
    def create_ml_dataset():
        """Trigger Cloud Function to build ML dataset."""
        import requests
        url = "https://us-central1-pipeline-882-team-project.cloudfunctions.net/create_ml_dataset"
        print("🚀 Triggering ML dataset creation Cloud Function...")
        response = requests.get(url)
        print(f"✅ ML dataset creation response: {response.text}")
        return response.text

    

    # --------------------------------------------------
    # Task Dependencies
    # --------------------------------------------------
    fred_extracts = extract_fred.expand(series_id=fred_series)
    fred_loads = load_fred_to_bq.expand(payload=fred_extracts)
    fred_landing = load_fred_landing(fred_loads)

    yf_extracts = extract_yfinance.expand(ticker=yfinance_tickers)
    yf_loads = load_yfinance_to_bq.expand(payload=yf_extracts)
    yf_landing = load_yfinance_landing(yf_loads)

    # Run ML dataset creation after both pipelines complete
    create_ml_task = create_ml_dataset()
    model_registration = register_model()
    register_ml_task = register_dataset()

    # Dependencies
    create_ml_task.set_upstream([fred_landing, yf_landing])
    register_ml_task.set_upstream([create_ml_task, model_registration])


# Instantiate DAG
credit_risk_pipeline()
