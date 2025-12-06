# dags/credit-risk-pipeline.py
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Airflow 2.10+ SDK (avoid deprecated warnings)
from airflow.sdk import dag, task, get_current_context
from airflow.exceptions import AirflowSkipException, AirflowFailException
from google.cloud import bigquery
from jinja2 import Template
import requests
import yaml
import json
import uuid
import math

# -----------------------------
# Project constants
# -----------------------------
PROJECT = "pipeline-882-team-project"
RAW_DATASET = "raw"
GOLD_DATASET = "gold"
MLOPS_DATASET = "mlops"
MODEL_ID = "credit_delinquency_model"
MODEL_NAME = "Credit Delinquency Rate Predictor"

# Cloud Function endpoints
CF_FETCH_FRED = f"https://us-central1-{PROJECT}.cloudfunctions.net/raw-fetch-fred-append"
CF_UPLOAD_FRED = f"https://us-central1-{PROJECT}.cloudfunctions.net/raw_upload_fred_append"
CF_LANDING_FRED = f"https://us-central1-{PROJECT}.cloudfunctions.net/landing-load-fred"
CF_FETCH_YF = f"https://us-central1-{PROJECT}.cloudfunctions.net/raw_fetch_yfinance_append"
CF_UPLOAD_YF = f"https://us-central1-{PROJECT}.cloudfunctions.net/raw_upload_yfinance_append"
CF_LANDING_YF = f"https://us-central1-{PROJECT}.cloudfunctions.net/landing_load_yfinance_append"
CF_CREATE_ML_DS = f"https://us-central1-{PROJECT}.cloudfunctions.net/create_ml_dataset"
CF_TRAIN_MODEL = f"https://us-central1-{PROJECT}.cloudfunctions.net/ml-train-model"
INFERENCE_ENDPOINT = f"https://us-central1-{PROJECT}.cloudfunctions.net/ml-predict-model"

# -----------------------------
# Helper functions
# -----------------------------
def read_file(path_str: str) -> str:
    """Read a text file (e.g., SQL template)."""
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path_str}")
    return p.read_text(encoding="utf-8")

def bq_client() -> bigquery.Client:
    """Create and return a BigQuery client."""
    return bigquery.Client(project=PROJECT)

def run_execute(sql: str):
    """Execute a SQL query and wait until it completes."""
    bq_client().query(sql).result()

def run_fetchone(sql: str):
    """Fetch a single row from a BigQuery query result."""
    rows = list(bq_client().query(sql).result())
    return rows[0] if rows else None

def invoke_function(url, params=None, method="GET", timeout=180):
    """Invoke a Cloud Function endpoint with basic error handling and JSON fallback."""
    params = params or {}
    try:
        if method.upper() == "POST":
            resp = requests.post(url, json=params, timeout=timeout)
        else:
            resp = requests.get(url, params=params, timeout=timeout)

        # Normalize skip conditions when no new data is available
        if resp.status_code == 204 or "no new" in resp.text.lower():
            raise AirflowSkipException("No new data.")

        if resp.status_code >= 400:
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

def json_sanitize(obj):
    """Recursively replace NaN/Inf with None for safe JSON serialization (XCom/BigQuery)."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    return obj

# -----------------------------
# DAG definition
# -----------------------------
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
    End-to-end MLOps pipeline for credit delinquency forecasting.
    Steps:
    1. Incremental ingestion (FRED + YFinance) and landing transformations
    2. Build ML dataset (GOLD layer)
    3. Register model and dataset metadata
    4. Train multiple model variants via Cloud Functions
    5. Record training runs in BigQuery
    6. Select best-performing model by sMAPE, version, and deploy (policy-based)
    """

    # -----------------------------
    # Config paths
    # -----------------------------
    FRED_CONFIG = "/usr/local/airflow/include/config/fred_series.yaml"
    YFIN_CONFIG = "/usr/local/airflow/include/config/yfinance_tickers.yaml"
    SQL_DIR = "/usr/local/airflow/include/sql"

    # Model configuration grid
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

    # -----------------------------
    # Load YAML configs
    # -----------------------------
    with open(FRED_CONFIG, "r") as f:
        fred_series = yaml.safe_load(f)["series"]

    with open(YFIN_CONFIG, "r") as f:
        yfinance_tickers = yaml.safe_load(f)["tickers"]

    # -----------------------------
    # FRED ingestion and landing
    # -----------------------------
    @task
    def extract_fred(series_id: str) -> dict:
        """Fetch incremental FRED series data via Cloud Function."""
        ctx = get_current_context()
        payload = {"series_id": series_id, "run_id": ctx["dag_run"].run_id}
        resp = invoke_function(CF_FETCH_FRED, params=payload)
        return {"series_id": series_id, "new_data": True, "resp": resp}

    @task
    def load_fred_to_bq(payload: dict):
        """Load fetched FRED data into BigQuery raw dataset."""
        sid = payload.get("series_id")
        if not payload.get("new_data"):
            raise AirflowSkipException(f"No new FRED data for {sid}")
        resp = invoke_function(CF_UPLOAD_FRED, params={"series_id": sid})
        return {"series_id": sid, "loaded": True, "resp": resp}

    @task(trigger_rule="all_done")
    def load_fred_landing(results: list):
        """Trigger FRED landing transformation after all loads complete."""
        if not any(r.get("loaded") for r in results if isinstance(r, dict)):
            raise AirflowSkipException("No FRED updates → skip landing.")
        return invoke_function(CF_LANDING_FRED)

    # -----------------------------
    # YFinance ingestion and landing
    # -----------------------------
    @task
    def extract_yfinance(ticker: str) -> dict:
        """Fetch incremental stock data from YFinance via Cloud Function."""
        ctx = get_current_context()
        payload = {"ticker": ticker, "run_id": ctx["dag_run"].run_id}
        resp = invoke_function(CF_FETCH_YF, params=payload)
        status = str(resp.get("status", "")).lower()
        if status in ["no_data", "skipped", "up_to_date"] or "no new" in str(resp).lower():
            raise AirflowSkipException(f"⏩ No new data for {ticker}")
        if "ticker" not in resp:
            resp["ticker"] = ticker
        return resp

    @task(retries=2, retry_delay=timedelta(seconds=20))
    def load_yfinance_to_bq(payload: dict) -> dict:
        """Upload YFinance data into BigQuery raw dataset."""
        ticker = payload.get("ticker")
        if not ticker:
            raise AirflowSkipException(f"No ticker in payload: {payload}")
        resp = invoke_function(CF_UPLOAD_YF, params={"ticker": ticker})
        return {"ticker": ticker, "status": "success", "resp": resp}

    @task(trigger_rule="all_done")
    def load_yfinance_landing(results: list):
        """Run YFinance landing transformation after all loads."""
        if not any(r.get("status") == "success" for r in results if isinstance(r, dict)):
            raise AirflowSkipException("No YFinance updates → skip landing.")
        return invoke_function(CF_LANDING_YF)

    # -----------------------------
    # Build ML dataset (GOLD layer)
    # -----------------------------
    @task(trigger_rule="all_done")
    def create_ml_dataset():
        """Trigger the ML dataset creation Cloud Function."""
        resp = invoke_function(CF_CREATE_ML_DS)
        print(f"✅ ML dataset creation response: {resp}")
        return resp

    # -----------------------------
    # Register model and dataset
    # -----------------------------
    @task
    def register_model():
        """Insert or update model metadata in the MLOps registry."""
        vals = {
            "model_id": MODEL_ID,
            "model_name": MODEL_NAME,
            "owner": "analytics_team",
            "business_problem": "Forecast weekly credit delinquency rates using macro & financial indicators",
            "ticket_number": "CR-001",
            "tags_json": json.dumps({"target": "delinquency_rate", "frequency": "weekly"}, ensure_ascii=False, allow_nan=False),
        }

        tpl_path = f"{SQL_DIR}/mlops-model-registry.sql"
        if Path(tpl_path).exists():
            sql = Template(read_file(tpl_path)).render(**vals)
        else:
            sql = f"""
            MERGE `{PROJECT}.{MLOPS_DATASET}.model` T
            USING (SELECT '{vals["model_id"]}' AS model_id) S
            ON T.model_id = S.model_id
            WHEN MATCHED THEN UPDATE SET
                model_name = '{vals["model_name"]}',
                owner = '{vals["owner"]}',
                business_problem= '{vals["business_problem"].replace("'", "''")}',
                ticket_number = '{vals["ticket_number"]}',
                tags_json = '{vals["tags_json"].replace("'", "''")}'
            WHEN NOT MATCHED THEN INSERT
                (model_id, model_name, owner, business_problem, ticket_number, tags_json, created_at)
            VALUES
                ('{vals["model_id"]}','{vals["model_name"]}','{vals["owner"]}',
                 '{vals["business_problem"].replace("'", "''")}','{vals["ticket_number"]}',
                 '{vals["tags_json"].replace("'", "''")}', CURRENT_TIMESTAMP());
            """

        run_execute(sql)
        print("✅ Model registered.")
        return vals

    @task
    def register_dataset():
        """Register dataset snapshot (row & feature counts)."""
        # Count rows
        r = run_fetchone(f"""
        SELECT COUNT(*) AS c
        FROM `{PROJECT}.{GOLD_DATASET}.fact_all_indicators_weekly`
        """)
        row_count = int(r["c"]) if r and "c" in r.keys() else (int(r[0]) if r else 0)

        # Count features from a single sample row (by counting JSON keys)
        f = run_fetchone(f"""
        SELECT COUNT(*) AS f
        FROM UNNEST(REGEXP_EXTRACT_ALL(
            TO_JSON_STRING((SELECT AS STRUCT * FROM `{PROJECT}.{GOLD_DATASET}.fact_all_indicators_weekly` LIMIT 1)),
            r'"[^"]*":'
        ))
        """)
        feature_count = int(f["f"]) if f and "f" in f.keys() else (int(f[0]) if f else 0)

        meta = {
            "dataset_id": f"ds_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}",
            "data_version": datetime.now().strftime("%Y-%m-%d"),
            "row_count": row_count,
            "feature_count": feature_count,
            "model_id": MODEL_ID,
        }

        tpl_path = f"{SQL_DIR}/mlops-dataset-registry.sql"
        if Path(tpl_path).exists():
            sql = Template(read_file(tpl_path)).render(**meta)
        else:
            sql = f"""
            INSERT INTO `{PROJECT}.{MLOPS_DATASET}.dataset`
            (dataset_id, model_id, data_version, row_count, feature_count, created_at)
            VALUES
            ('{meta["dataset_id"]}','{meta["model_id"]}','{meta["data_version"]}',
             {meta["row_count"]},{meta["feature_count"]}, CURRENT_TIMESTAMP())
            """

        run_execute(sql)
        print("✅ Dataset registered.")
        return meta

    # -----------------------------
    # Train and persist runs
    # -----------------------------
    @task(
        retries=0,
        max_active_tis_per_dag=2,  # Limit concurrency for stability
    )
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
        
        try:
            # Cloud Function ONLY accepts GET style query-string params
            result = invoke_function(
                CF_TRAIN_MODEL,
                params=params,
                method="GET"  # IMPORTANT: CF reads request.args, not JSON body
            )
            
            # Expected shape returned by CF:
            # { run_id, algorithm, best_hyperparameters, gcs_path, metrics, feature_count }
            
            # # Ensure status is set to 'completed' for successful runs
            # if 'status' not in result:
            #     result['status'] = 'completed'
            
            return result
        except Exception as e:
            print(f"❌ Training failed for {cfg['algorithm']}: {e}")
            # Return a failure marker that downstream tasks can handle
            return {
                "run_id": run_id,
                "algorithm": cfg["algorithm"],
                "status": "failed",
                "error": str(e),
                "dataset_id": ds_meta["dataset_id"]
            }

    @task(trigger_rule="all_done")
    def register_training_run(model_result: dict):
        """Persist a single training run (params/metrics/artifact) into MLOps tables."""
        # Skip if training failed or returned invalid result
        if not isinstance(model_result, dict) or "run_id" not in model_result:
            raise AirflowSkipException("Training task failed or returned invalid result")
        
        # Skip if the training explicitly failed
        if model_result.get("status") == "failed":
            print(f"⚠️ Skipping failed training run: {model_result.get('algorithm', 'unknown')}")
            raise AirflowSkipException(f"Training failed: {model_result.get('error', 'unknown error')}")
        
        params_safe = json_sanitize(model_result.get("params", {}))
        metrics_safe = json_sanitize(model_result.get("metrics", {}))
        params_json = json.dumps(params_safe, ensure_ascii=False, allow_nan=False).replace("'", "''")
        metrics_json = json.dumps(metrics_safe, ensure_ascii=False, allow_nan=False).replace("'", "''")
        artifact_path = (model_result.get("artifact", model_result.get("gcs_path","")) or "").replace("'", "''")
        status_val = model_result.get("status", "completed")
        
        sql = f"""
        INSERT INTO `{PROJECT}.{MLOPS_DATASET}.training_run`
        (run_id, model_id, dataset_id, params, metrics, artifact, status, created_at)
        VALUES (
            '{model_result["run_id"]}',
            '{MODEL_ID}',
            '{model_result.get("dataset_id","")}',
            '{params_json}',
            '{metrics_json}',
            '{artifact_path}',
            '{status_val}',
            CURRENT_TIMESTAMP()
        )
        """
        
        run_execute(sql)
        print(f"✅ Training run recorded: {model_result['run_id']}")
        return {"run_id": model_result["run_id"], "dataset_id": model_result.get("dataset_id", "")}

    # -----------------------------
    # Select best, version, deploy
    # -----------------------------
    @task(trigger_rule="none_failed_min_one_success")
    def find_best_model(ds_meta: dict):
        """Find the best completed run by lowest sMAPE and fetch optional artifact path."""
        # Select best run by sMAPE
        best_sql = f"""
        SELECT run_id, params, metrics, artifact
        FROM `{PROJECT}.{MLOPS_DATASET}.training_run`
        WHERE model_id = '{MODEL_ID}'
          AND dataset_id = '{ds_meta["dataset_id"]}'
          AND status = 'completed'
        ORDER BY CAST(JSON_VALUE(metrics, '$.smape') AS FLOAT64) ASC
        LIMIT 1
        """
        
        best = run_fetchone(best_sql)
        if not best:
            raise AirflowFailException("No completed training runs found - all training failed.")
        
        def _get(row, key, idx):
            try:
                return row[key]
            except Exception:
                return row[idx]
        
        run_id = _get(best, "run_id", 0)
        params_json = _get(best, "params", 1)
        params = json.loads(params_json)
        metrics_json = _get(best, "metrics", 2)
        metrics = json.loads(metrics_json)
        artifact = _get(best, "artifact", 3)

        algorithm = params.get('algorithm', 'N/A')

        best_smape = metrics.get("smape")

        print(f"Best Model Found: run_id = {run_id}")
        print(f"Algorithm: {algorithm}")
        print(f"Metrics - SMAPE: {best_smape}")
        print(f"Artifact: {artifact}")

        # # Locate baseline sMAPE for comparison
        # base_sql = f"""
        # SELECT JSON_VALUE(metrics, '$.smape') AS base_smape
        # FROM `{PROJECT}.{MLOPS_DATASET}.training_run`
        # WHERE model_id = '{MODEL_ID}'
        #   AND dataset_id = '{ds_meta["dataset_id"]}'
        #   AND status = 'completed'
        #   AND JSON_VALUE(params, '$.algorithm') = 'base'
        # ORDER BY created_at DESC
        # LIMIT 1
        # """
        
        # base = run_fetchone(base_sql)
        # base_smape = None
        # if base:
        #     try:
        #         base_smape = float(base["base_smape"]) if base["base_smape"] is not None else None
        #     except Exception:
        #         base_smape = float(base[0]) if base[0] is not None else None


        # # # Check if 'artifact' column exists (some environments may omit it)
        # # chk = run_fetchone(f"""
        # # SELECT COUNT(*) AS c
        # # FROM `{PROJECT}.{MLOPS_DATASET}`.INFORMATION_SCHEMA.COLUMNS
        # # WHERE table_name = 'training_run' AND column_name = 'artifact'
        # # """)
        
        # # has_artifact_col = False
        # # try:
        # #     has_artifact_col = (int(chk["c"]) if chk and "c" in chk.keys() else int(chk[0])) > 0
        # # except Exception:
        # #     has_artifact_col = False

        # artifact: Optional[str] = ""
        # if has_artifact_col:
        #     art_row = run_fetchone(f"""
        #     SELECT artifact
        #     FROM `{PROJECT}.{MLOPS_DATASET}.training_run`
        #     WHERE run_id = '{run_id}'
        #     LIMIT 1
        #     """)
        #     if art_row:
        #         try:
        #             artifact = art_row["artifact"]
        #         except Exception:
        #             artifact = art_row[0]
        
        # artifact = artifact or ""

        return {
            "run_id": run_id,
            "params": params,
            "metrics": metrics,
            "artifact": artifact,  # empty string if unavailable
            # "best_smape": best_smape,
            "dataset_id": ds_meta["dataset_id"],
        }

    @task
    def register_model_version(best: dict):
        """Create a new model version with status based on improvement over baseline."""
        def to_float_safe(x):
            try:
                v = float(x)
                if math.isnan(v) or math.isinf(v):
                    return None
                return v
            except Exception:
                return None

        # Locate baseline sMAPE for comparison
        base_sql = f"""
        SELECT JSON_VALUE(metrics, '$.smape') AS base_smape
        FROM `{PROJECT}.{MLOPS_DATASET}.deployment`
        WHERE traffic_split = 1.0
        ORDER BY deployed_at DESC
        LIMIT 1
        """
        
        base = run_fetchone(base_sql)
        base_smape = None
        if base:
            try:
                base_smape = float(base["base_smape"]) if base["base_smape"] is not None else None
            except Exception:
                base_smape = float(base[0]) if base[0] is not None else None


        base = to_float_safe(base_smape)
        new = to_float_safe(best.get("metrics", {}).get("smape"))

        # Conservative policy: mark as candidate unless >=10% improvement over a valid baseline
        status = "candidate"
        # if base is not None and base > 0 and new is not None:
        #     improvement = (base - new) / base
        #     status = "approved" if improvement >= 0.01 else "candidate"
        #     print(f"Baseline sMAPE={base:.6f}, new sMAPE={new:.6f}, improvement={improvement:.2%}")
        # else:
        #     print(f"⚠️ Baseline or new sMAPE invalid (baseline={base}, new={new}); marking as candidate.")

        if base is None:
            status = "approved" 
        else:
            improvement = (base - new) / base
            if improvement>= 0.01:
                status = "approved" 

        model_version_id = f"{MODEL_ID}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        artifact_path = (best.get("artifact") or "").replace("'", "''")
        metrics_json = json.dumps(json_sanitize(best.get("metrics", {})), ensure_ascii=False, allow_nan=False).replace("'", "''")

        insert_sql = f"""
        INSERT INTO `{PROJECT}.{MLOPS_DATASET}.model_version`
        (model_version_id, model_id, training_run_id, artifact_path, metrics_json, status, created_at)
        VALUES (
            '{model_version_id}',
            '{MODEL_ID}',
            '{best["run_id"]}',
            '{artifact_path}',
            '{metrics_json}',
            '{status}',
            CURRENT_TIMESTAMP()
        )
        """
        
        run_execute(insert_sql)
        print(f"✅ Model version recorded: {model_version_id} ({status})")
        return {
                "model_version_id": model_version_id, 
                "status": status, 
                "model_id": MODEL_ID, 
                "metrics_json": metrics_json, 
                "artifact_path": artifact_path
                }

    @task
    def register_deployment(mv: dict):
        """
        Record a deployment row. If status is 'approved', route full traffic (1.0).
        Otherwise, stage the version with traffic=0.0 for auditing and later promotion.
        """
        model_version_id = mv['model_version_id']
        is_approved = (mv.get("status") == "approved")
        
        # Generate deployment ID
        deployment_id = f"deploy_{model_version_id}"
        
        # Determine traffic split based on approval status
        traffic = 1.0 if is_approved else 0.0
        
        if is_approved:
            # Archive existing active deployments for this model (set traffic to 0.0)
            # This ensures only the newest approved deployment is active
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
            
            print("📦 Archiving old deployments...")
            print(archive_sql)
            run_execute(archive_sql)

            # Always insert a new deployment row (approved -> 1.0 traffic, candidate -> 0.0 traffic)
            insert_sql = f"""
            INSERT INTO `{PROJECT}.{MLOPS_DATASET}.deployment`
            (deployment_id, model_version_id, endpoint_url, traffic_split, deployed_at, params, metrics, artifact)
            VALUES (
                '{deployment_id}',
                '{model_version_id}',
                '{INFERENCE_ENDPOINT}',
                {traffic},
                CURRENT_TIMESTAMP(), 
                'N/A', 
                '{metrics_json}', 
                '{artifact_path}'
            )
            """
            
            print("📝 Registering new deployment...")
            print(insert_sql)
            run_execute(insert_sql)

        if is_approved:
            print(f"🚀 Deployed: {deployment_id} → {INFERENCE_ENDPOINT} (traffic=1.0)")
            print(f"✅ Model version {model_version_id} is now serving production traffic")
        else:
            print(f"📝 Staged (not approved): {deployment_id} recorded with traffic=0.0")
            print(f"⚠️  Model version {model_version_id} is registered but not serving traffic")

        return {
            "deployed": is_approved,
            "deployment_id": deployment_id,
            "model_version_id": model_version_id,
            "endpoint_url": INFERENCE_ENDPOINT,
            "traffic_split": traffic
        }

    # -----------------------------
    # Orchestration
    # -----------------------------

    # Ingestion flows
    fred_extracts = extract_fred.expand(series_id=fred_series)
    fred_loads = load_fred_to_bq.expand(payload=fred_extracts)
    fred_land = load_fred_landing(fred_loads)

    yf_extracts = extract_yfinance.expand(ticker=yfinance_tickers)
    yf_loads = load_yfinance_to_bq.expand(payload=yf_extracts)
    yf_land = load_yfinance_landing(yf_loads)

    # Build GOLD dataset after both landings complete
    ml_ds = create_ml_dataset()
    ml_ds.set_upstream([fred_land, yf_land])

    # Register model & dataset snapshot
    model_reg = register_model()
    dataset_reg = register_dataset()
    dataset_reg.set_upstream([ml_ds, model_reg])

    # Train models in parallel (with concurrency limits)
    train_results = train_model.partial(ds_meta=dataset_reg).expand(cfg=model_configs)

    # Persist training runs (with concurrency limits) - proceed if at least one training succeeds
    recorded_runs = register_training_run.expand(model_result=train_results)

    # Select best model after runs are recorded - proceed if at least one was recorded successfully
    best = find_best_model(dataset_reg)
    best.set_upstream(recorded_runs)

    # Version the best model, then register a deployment row (approved → traffic=1.0, else 0.0)
    mv = register_model_version(best)
    register_deployment(mv)


# Instantiate DAG
credit_risk_pipeline()