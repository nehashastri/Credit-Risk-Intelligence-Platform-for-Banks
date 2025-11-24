# dags/credit-risk-pipeline.py
from airflow.decorators import dag, task
from datetime import datetime, timedelta
from airflow.operators.python import get_current_context
from airflow.exceptions import AirflowSkipException
import requests, yaml, json, time, uuid
from jinja2 import Template
import utils

# --------------------------------------------------
# Utility: Safe Cloud Function invoker
# --------------------------------------------------
def invoke_function(url, params={}, method="GET"):
    """Unified Cloud Function invoker with error handling and JSON fallback."""
    try:
        if method.upper() == "POST":
            resp = requests.post(url, json=params)
        else:
            resp = requests.get(url, params=params)

        # Normalize common skip/empty responses
        if resp.status_code == 204 or "no new" in resp.text.lower():
            raise AirflowSkipException("No new data.")
        elif resp.status_code == 500:
            raise AirflowSkipException("Server 500 error.")
        elif resp.status_code >= 400:
            print(f"❌ Request failed ({resp.status_code}): {resp.text}")
            resp.raise_for_status()

        # Try JSON first, gracefully fallback to text
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
    tags=["credit-risk", "fred", "yfinance", "ml-dataset", "mlops"],
)
def credit_risk_pipeline():
    """
    Credit Risk Pipeline + MLOps (Baseline Rolling Mean + RF/XGB/LGBM/ElasticNet)

    Stages:
      1) Incremental ingest (FRED & YFinance) → Landing transforms
      2) Build ML dataset
      3) Register model + dataset snapshot (MLOps registry tables)
      4) Train models in parallel via Cloud Function
      5) Persist training runs (params/metrics/artifacts)
      6) Select best (by sMAPE), register model version, deploy if approved
    """

    # --------------------------------------------------
    # Config / Constants
    # --------------------------------------------------
    FRED_CONFIG = "/usr/local/airflow/include/config/fred_series.yaml"
    YFIN_CONFIG = "/usr/local/airflow/include/config/yfinance_tickers.yaml"
    SQL_DIR     = "/usr/local/airflow/include/sql"

    PROJECT        = "pipeline-882-team-project"
    RAW_DATASET    = "raw"
    GOLD_DATASET   = "gold"
    MLOPS_DATASET  = "mlops"

    MODEL_ID   = "credit_delinquency_model"
    MODEL_NAME = "Credit Delinquency Rate Predictor"

    # Cloud Functions (ingest / transform / ML)
    CF_FETCH_FRED      = f"https://us-central1-{PROJECT}.cloudfunctions.net/raw-fetch-fred-append"
    CF_UPLOAD_FRED     = f"https://us-central1-{PROJECT}.cloudfunctions.net/raw_upload_fred_append"
    CF_LANDING_FRED    = f"https://us-central1-{PROJECT}.cloudfunctions.net/landing-load-fred"

    CF_FETCH_YF        = f"https://us-central1-{PROJECT}.cloudfunctions.net/raw_fetch_yfinance_append"
    CF_UPLOAD_YF       = f"https://us-central1-{PROJECT}.cloudfunctions.net/raw_upload_yfinance_append"
    CF_LANDING_YF      = f"https://us-central1-{PROJECT}.cloudfunctions.net/landing_load_yfinance_append"

    CF_CREATE_ML_DS    = f"https://us-central1-{PROJECT}.cloudfunctions.net/create_ml_dataset"
    # Training CF endpoint provided by you (supports base / linear_regression / elastic_net / random_forest / gradient_boosting / xgboost / lightgbm)
    CF_TRAIN_MODEL     = f"https://us-central1-{PROJECT}.cloudfunctions.net/train-credit-model"

    # Placeholder inference endpoint (fill with your actual predictor endpoint when ready)
    INFERENCE_ENDPOINT = f"https://us-central1-{PROJECT}.cloudfunctions.net/ml_predict_credit"

    # --------------------------------------------------
    # Model configs (Baseline = rolling mean "base" + 4 ML models)
    #   NOTE: The CF expects "base" as the baseline algorithm name.
    # --------------------------------------------------
    model_configs = [
    {"algorithm": "base"},
    {"algorithm": "random_forest"},
    {"algorithm": "xgboost"},
    {"algorithm": "lightgbm"},
    {"algorithm": "elastic_net"},
    {"algorithm": "gradient_boosting"},
    {"algorithm": "arima"},
    {"algorithm": "sarimax"},
    {"algorithm": "varmax"},
]


    # --------------------------------------------------
    # Load YAML configs
    # --------------------------------------------------
    with open(FRED_CONFIG, "r") as f:
        fred_series = yaml.safe_load(f)["series"]

    with open(YFIN_CONFIG, "r") as f:
        yfinance_tickers = yaml.safe_load(f)["tickers"]

    # --------------------------------------------------
    # FRED: Extract → Upload → Landing
    # --------------------------------------------------
    @task
    def extract_fred(series_id: str) -> dict:
        """Fetch the latest FRED data for a series and stage to GCS."""
        ctx = get_current_context()
        payload = {"series_id": series_id, "run_id": ctx["dag_run"].run_id}
        resp = invoke_function(CF_FETCH_FRED, params=payload)
        return {"series_id": series_id, "new_data": True, "resp": resp}

    @task
    def load_fred_to_bq(payload: dict):
        """Load appended FRED file(s) from GCS into BigQuery RAW tables."""
        series_id = payload.get("series_id")
        if not payload.get("new_data"):
            raise AirflowSkipException(f"No new FRED data for {series_id}")
        resp = invoke_function(CF_UPLOAD_FRED, params={"series_id": series_id})
        return {"series_id": series_id, "loaded": True, "resp": resp}

    @task(trigger_rule="all_done")
    def load_fred_landing(results: list):
        """Run FRED landing transform only if any series was updated."""
        if not any(r.get("loaded") for r in results if isinstance(r, dict)):
            raise AirflowSkipException("No FRED updates → skip landing.")
        return invoke_function(CF_LANDING_FRED)

    # --------------------------------------------------
    # YFinance: Extract → Upload → Landing
    # --------------------------------------------------
    @task
    def extract_yfinance(ticker: str) -> dict:
        """Fetch the latest YFinance data for a ticker and stage to GCS."""
        ctx = get_current_context()
        payload = {"ticker": ticker, "run_id": ctx["dag_run"].run_id}
        resp = invoke_function(CF_FETCH_YF, params=payload)
        status = str(resp.get("status", "")).lower()
        if status in ["no_data", "skipped", "up_to_date"] or "no new" in str(resp).lower():
            raise AirflowSkipException(f"⏩ No new data for {ticker}")
        return resp

    @task(retries=2, retry_delay=timedelta(seconds=20))
    def load_yfinance_to_bq(payload: dict) -> dict:
        """Load appended YFinance file(s) from GCS into BigQuery RAW tables."""
        ticker = payload.get("ticker")
        if not ticker:
            raise AirflowSkipException(f"No ticker in payload: {payload}")
        resp = invoke_function(CF_UPLOAD_YF, params={"ticker": ticker})
        return {"ticker": ticker, "status": "success", "resp": resp}

    @task(trigger_rule="all_done")
    def load_yfinance_landing(results: list):
        """Run YFinance landing transform only if any ticker was updated."""
        if not any(r.get("status") == "success" for r in results if isinstance(r, dict)):
            raise AirflowSkipException("No YFinance updates → skip landing.")
        return invoke_function(CF_LANDING_YF)

    # --------------------------------------------------
    # Build ML Dataset
    # --------------------------------------------------
    @task(trigger_rule="all_done")
    def create_ml_dataset():
        """Trigger CF that materializes ML-ready GOLD dataset/features."""
        resp = invoke_function(CF_CREATE_ML_DS)
        print(f"✅ ML dataset creation response: {resp}")
        return resp

    # --------------------------------------------------
    # MLOps: Register Model & Dataset snapshot
    # --------------------------------------------------
    @task
    def register_model():
        """Insert/Upsert model metadata into mlops.model."""
        model_vals = {
            "model_id": MODEL_ID,
            "model_name": MODEL_NAME,
            "owner": "analytics_team",
            "business_problem": "Forecast weekly credit delinquency rates using macro & financial indicators",
            "ticket_number": "CR-001",
            "tags_json": json.dumps({"target": "delinquency_rate", "frequency": "weekly"})
        }
        s = utils.read_sql(f"{SQL_DIR}/mlops-model-registry.sql")
        sql = Template(s).render(**model_vals)
        utils.run_execute(sql)
        print("✅ Model registered.")
        return model_vals

    @task
    def register_dataset():
        """Insert dataset snapshot metadata into mlops.dataset."""
        row_count = utils.run_fetchone(f"""
            SELECT COUNT(*) 
            FROM `{PROJECT}.{GOLD_DATASET}.fact_all_indicators_weekly`
        """)[0]

        feature_count = utils.run_fetchone(f"""
            SELECT COUNT(*) 
            FROM UNNEST(REGEXP_EXTRACT_ALL(
                TO_JSON_STRING((SELECT AS STRUCT * 
                                FROM `{PROJECT}.{GOLD_DATASET}.fact_all_indicators_weekly` 
                                LIMIT 1)),
                r'"[^"]*":'
            ))
        """)[0]

        dataset_metadata = {
            "dataset_id": f"ds_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}",
            "data_version": datetime.now().strftime("%Y-%m-%d"),
            "row_count": row_count,
            "feature_count": feature_count,
            "model_id": MODEL_ID,
        }

        s = utils.read_sql(f"{SQL_DIR}/mlops-dataset-registry.sql")
        sql = Template(s).render(**dataset_metadata)
        utils.run_execute(sql)
        print("✅ Dataset registered.")
        return dataset_metadata

    # --------------------------------------------------
    # Train models (Baseline + 4) and persist training run
    # --------------------------------------------------
    @task
    def train_model(cfg: dict, ds_meta: dict):
        """Call CF to train one model; return full training result payload."""
        
        run_id = f"run_{MODEL_ID}_{cfg['algorithm']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Build query params EXPECTED by Cloud Function
        params = {
            "run_id": run_id,
            "model_id": MODEL_ID,
            "algorithm": cfg["algorithm"],
            "dataset_id": ds_meta["dataset_id"],
        }

        print(f"🚀 Training {cfg['algorithm']} ...")

        # Cloud Function ONLY accepts GET style query-string params
        result = invoke_function(
            CF_TRAIN_MODEL,
            params=params,
            method="GET"    # IMPORTANT: CF reads request.args, not JSON body
        )

        # Expected shape returned by CF: 
        # { run_id, algorithm, best_hyperparameters, gcs_path, metrics, feature_count }
        return result


    @task
    def register_training_run(model_result: dict):
        """Insert one training run record into mlops.training_run."""
        sql = f"""
        INSERT INTO `{PROJECT}.{MLOPS_DATASET}.training_run`
        (run_id, model_id, dataset_id, params, metrics, artifact, status, created_at)
        VALUES (
            '{model_result["run_id"]}',
            '{MODEL_ID}',
            '{model_result["dataset_id"]}',
            '{json.dumps(model_result.get("params", {}))}',
            '{json.dumps(model_result.get("metrics", {}))}',
            '{model_result.get("artifact", "")}',
            '{model_result.get("status", "completed")}',
            CURRENT_TIMESTAMP()
        )
        """
        utils.run_execute(sql)
        print(f"✅ Training run recorded: {model_result['run_id']}")
        return {"run_id": model_result["run_id"], "dataset_id": model_result["dataset_id"]}

    # --------------------------------------------------
    # Select best (by sMAPE), register version, deploy if approved
    # --------------------------------------------------
    @task
    def find_best_model(ds_meta: dict):
        """Pick the best completed run for this dataset by lowest sMAPE."""
        best_sql = f"""
        SELECT run_id, params, metrics, artifact
        FROM `{PROJECT}.{MLOPS_DATASET}.training_run`
        WHERE model_id = '{MODEL_ID}'
          AND dataset_id = '{ds_meta["dataset_id"]}'
          AND status = 'completed'
        ORDER BY CAST(JSON_VALUE(metrics, '$.portfolio.smape') AS FLOAT64) ASC
        LIMIT 1
        """
        best = utils.run_fetchone(best_sql)
        if not best:
            raise AirflowSkipException("No completed training runs found.")

        # Fetch the latest baseline's sMAPE for relative improvement checks
        base_sql = f"""
        SELECT JSON_VALUE(metrics, '$.portfolio.smape') AS base_smape
        FROM `{PROJECT}.{MLOPS_DATASET}.training_run`
        WHERE model_id = '{MODEL_ID}'
          AND dataset_id = '{ds_meta["dataset_id"]}'
          AND status = 'completed'
          AND JSON_VALUE(params, '$.algorithm') = 'base'
        ORDER BY created_at DESC
        LIMIT 1
        """
        base = utils.run_fetchone(base_sql)
        base_smape = float(base[0]) if base and base[0] is not None else None

        return {
            "run_id": best[0],
            "params": json.loads(best[1]),
            "metrics": json.loads(best[2]),
            "artifact": best[3],
            "baseline_smape": base_smape,
            "dataset_id": ds_meta["dataset_id"]
        }

    @task
    def register_model_version(best: dict):
        """Insert a model version; approve if improved >=10% over baseline sMAPE."""
        status = "approved"
        if best.get("baseline_smape") is not None:
            new_smape = float(best["metrics"]["portfolio"]["smape"])
            improvement = (best["baseline_smape"] - new_smape) / best["baseline_smape"]
            status = "approved" if improvement >= 0.10 else "candidate"
            print(f"Baseline sMAPE={best['baseline_smape']:.4f}, new sMAPE={new_smape:.4f}, improvement={improvement:.2%}")
        else:
            print("No baseline found; approving first model version by default.")

        model_version_id = f"{MODEL_ID}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        insert_sql = f"""
        INSERT INTO `{PROJECT}.{MLOPS_DATASET}.model_version`
        (model_version_id, model_id, training_run_id, artifact_path, metrics_json, status, created_at)
        VALUES (
            '{model_version_id}',
            '{MODEL_ID}',
            '{best["run_id"]}',
            '{best["artifact"].replace("'", "''")}',
            '{json.dumps(best["metrics"]).replace("'", "''")}',
            '{status}',
            CURRENT_TIMESTAMP()
        )
        """
        utils.run_execute(insert_sql)
        print(f"✅ Model version recorded: {model_version_id} ({status})")
        return {"model_version_id": model_version_id, "status": status}

    @task
    def register_deployment(mv: dict):
        """Register deployment (switch 100% traffic to newest approved version)."""
        if mv["status"] != "approved":
            print("Deployment skipped (status != approved).")
            return {"deployed": False, "model_version_id": mv["model_version_id"]}

        # Archive prior deployments for this model (set traffic to 0%)
        archive_sql = f"""
        UPDATE `{PROJECT}.{MLOPS_DATASET}.deployment`
        SET traffic_split = 0.0
        WHERE deployment_id IN (
            SELECT d.deployment_id
            FROM `{PROJECT}.{MLOPS_DATASET}.deployment` d
            JOIN `{PROJECT}.{MLOPS_DATASET}.model_version` mv
              ON d.model_version_id = mv.model_version_id
            WHERE mv.model_id = '{MODEL_ID}'
              AND d.traffic_split > 0
        )
        """
        utils.run_execute(archive_sql)

        deployment_id = f"deploy_{mv['model_version_id']}"
        insert_sql = f"""
        INSERT INTO `{PROJECT}.{MLOPS_DATASET}.deployment`
        (deployment_id, model_version_id, endpoint_url, traffic_split, deployed_at)
        VALUES (
            '{deployment_id}',
            '{mv["model_version_id"]}',
            '{INFERENCE_ENDPOINT}',
            1.0,
            CURRENT_TIMESTAMP()
        )
        """
        utils.run_execute(insert_sql)
        print(f"🚀 Deployed: {deployment_id} → {INFERENCE_ENDPOINT}")
        return {"deployed": True, "deployment_id": deployment_id, "endpoint_url": INFERENCE_ENDPOINT}

    # --------------------------------------------------
    # Orchestration
    # --------------------------------------------------
    # Ingest: FRED
    fred_extracts  = extract_fred.expand(series_id=fred_series)
    fred_loads     = load_fred_to_bq.expand(payload=fred_extracts)
    fred_landing   = load_fred_landing(fred_loads)

    # Ingest: YFinance
    yf_extracts    = extract_yfinance.expand(ticker=yfinance_tickers)
    yf_loads       = load_yfinance_to_bq.expand(payload=yf_extracts)
    yf_landing     = load_yfinance_landing(yf_loads)

    # Build ML dataset after both landing transforms finish
    ml_ds_task     = create_ml_dataset()
    ml_ds_task.set_upstream([fred_landing, yf_landing])

    # Register model + dataset snapshot
    model_reg      = register_model()
    dataset_reg    = register_dataset()
    dataset_reg.set_upstream([ml_ds_task, model_reg])

    # Train all models in parallel and record runs
    train_results  = train_model.partial(ds_meta=dataset_reg).expand(cfg=model_configs)
    recorded_runs  = register_training_run.expand(model_result=train_results)

    # Select best, version it, and deploy if approved
    best_model     = find_best_model(dataset_reg)
    model_version  = register_model_version(best_model)
    register_deployment(model_version)


# Instantiate DAG
credit_risk_pipeline()