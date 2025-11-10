# dags/credit-risk-pipeline.py
from airflow.decorators import dag, task
from datetime import datetime, timedelta
from airflow.operators.python import get_current_context
from airflow.exceptions import AirflowSkipException
from google.cloud import bigquery
import requests, yaml, json, uuid
from jinja2 import Template
from pathlib import Path

# -----------------------------
# Project constants
# -----------------------------
PROJECT = "pipeline-882-team-project"
RAW_DATASET   = "raw"
GOLD_DATASET  = "gold"
MLOPS_DATASET = "mlops"

MODEL_ID   = "credit_delinquency_model"
MODEL_NAME = "Credit Delinquency Rate Predictor"

# Cloud Functions (ingest / transform / dataset / training)
CF_FETCH_FRED   = f"https://us-central1-{PROJECT}.cloudfunctions.net/raw-fetch-fred-append"
CF_UPLOAD_FRED  = f"https://us-central1-{PROJECT}.cloudfunctions.net/raw_upload_fred_append"
CF_LANDING_FRED = f"https://us-central1-{PROJECT}.cloudfunctions.net/landing-load-fred"

CF_FETCH_YF     = f"https://us-central1-{PROJECT}.cloudfunctions.net/raw_fetch_yfinance_append"
CF_UPLOAD_YF    = f"https://us-central1-{PROJECT}.cloudfunctions.net/raw_upload_yfinance_append"
CF_LANDING_YF   = f"https://us-central1-{PROJECT}.cloudfunctions.net/landing_load_yfinance_append"

CF_CREATE_ML_DS = f"https://us-central1-{PROJECT}.cloudfunctions.net/create_ml_dataset"
CF_TRAIN_MODEL  = f"https://us-central1-{PROJECT}.cloudfunctions.net/train-credit-model"  # expects querystring params

INFERENCE_ENDPOINT = f"https://us-central1-{PROJECT}.cloudfunctions.net/ml_predict_credit"  # placeholder

# -----------------------------
# Local helpers (replace external utils)
# -----------------------------
def read_file(path_str: str) -> str:
    """Read a local text file (e.g., SQL template)."""
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path_str}")
    return p.read_text(encoding="utf-8")

def bq_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)

def run_execute(sql: str):
    """Execute a BigQuery query and wait for completion."""
    bq_client().query(sql).result()

def run_fetchone(sql: str):
    """Return first row of a BigQuery query (Row object or None)."""
    rows = list(bq_client().query(sql).result())
    return rows[0] if rows else None

def invoke_function(url, params=None, method="GET", timeout=180):
    """HTTP invoker with error handling and JSON fallback."""
    params = params or {}
    try:
        if method.upper() == "POST":
            resp = requests.post(url, json=params, timeout=timeout)
        else:
            resp = requests.get(url, params=params, timeout=timeout)

        # normalize skip conditions
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

# -----------------------------
# DAG
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
    End-to-end credit risk pipeline with MLOps:
      1) Incremental ingest (FRED, YFinance) → landing transforms
      2) Build ML dataset (GOLD)
      3) Register model and dataset snapshot
      4) Train models via Cloud Function (baseline + ML)
      5) Record training runs
      6) Select best by sMAPE, register model version, deploy if approved
    """

    # -----------------------------
    # Config files
    # -----------------------------
    FRED_CONFIG = "/usr/local/airflow/include/config/fred_series.yaml"
    YFIN_CONFIG = "/usr/local/airflow/include/config/yfinance_tickers.yaml"
    SQL_DIR     = "/usr/local/airflow/include/sql"

    # Model grid (baseline + 4 ML)
    model_configs = [
        {"algorithm": "base",           "hyperparameters": {"type": "rolling_mean", "window": 6}},
        {"algorithm": "random_forest",  "hyperparameters": {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5}},
        {"algorithm": "xgboost",        "hyperparameters": {"max_depth": 4, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}},
        {"algorithm": "lightgbm",       "hyperparameters": {"num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.9}},
        {"algorithm": "elastic_net",    "hyperparameters": {"alpha": 0.5, "l1_ratio": 0.2}},
    ]

    # -----------------------------
    # Load YAML configs
    # -----------------------------
    with open(FRED_CONFIG, "r") as f:
        fred_series = yaml.safe_load(f)["series"]
    with open(YFIN_CONFIG, "r") as f:
        yfinance_tickers = yaml.safe_load(f)["tickers"]

    # -----------------------------
    # FRED: Extract → Upload → Landing
    # -----------------------------
    @task
    def extract_fred(series_id: str) -> dict:
        ctx = get_current_context()
        payload = {"series_id": series_id, "run_id": ctx["dag_run"].run_id}
        resp = invoke_function(CF_FETCH_FRED, params=payload)
        return {"series_id": series_id, "new_data": True, "resp": resp}

    @task
    def load_fred_to_bq(payload: dict):
        sid = payload.get("series_id")
        if not payload.get("new_data"):
            raise AirflowSkipException(f"No new FRED data for {sid}")
        resp = invoke_function(CF_UPLOAD_FRED, params={"series_id": sid})
        return {"series_id": sid, "loaded": True, "resp": resp}

    @task(trigger_rule="all_done")
    def load_fred_landing(results: list):
        if not any(r.get("loaded") for r in results if isinstance(r, dict)):
            raise AirflowSkipException("No FRED updates → skip landing.")
        return invoke_function(CF_LANDING_FRED)

    # -----------------------------
    # YFinance: Extract → Upload → Landing
    # -----------------------------
    @task
    def extract_yfinance(ticker: str) -> dict:
        ctx = get_current_context()
        payload = {"ticker": ticker, "run_id": ctx["dag_run"].run_id}
        resp = invoke_function(CF_FETCH_YF, params=payload)
        status = str(resp.get("status", "")).lower()
        if status in ["no_data", "skipped", "up_to_date"] or "no new" in str(resp).lower():
            raise AirflowSkipException(f"⏩ No new data for {ticker}")
        return resp  # should include {"ticker": ...}

    @task(retries=2, retry_delay=timedelta(seconds=20))
    def load_yfinance_to_bq(payload: dict) -> dict:
        ticker = payload.get("ticker")
        if not ticker:
            raise AirflowSkipException(f"No ticker in payload: {payload}")
        resp = invoke_function(CF_UPLOAD_YF, params={"ticker": ticker})
        return {"ticker": ticker, "status": "success", "resp": resp}

    @task(trigger_rule="all_done")
    def load_yfinance_landing(results: list):
        if not any(r.get("status") == "success" for r in results if isinstance(r, dict)):
            raise AirflowSkipException("No YFinance updates → skip landing.")
        return invoke_function(CF_LANDING_YF)

    # -----------------------------
    # Build ML dataset (GOLD)
    # -----------------------------
    @task(trigger_rule="all_done")
    def create_ml_dataset():
        resp = invoke_function(CF_CREATE_ML_DS)
        print(f"✅ ML dataset creation response: {resp}")
        return resp

    # -----------------------------
    # MLOps: register model & dataset
    # -----------------------------
    @task
    def register_model():
        vals = {
            "model_id": MODEL_ID,
            "model_name": MODEL_NAME,
            "owner": "analytics_team",
            "business_problem": "Forecast weekly credit delinquency rates using macro & financial indicators",
            "ticket_number": "CR-001",
            "tags_json": json.dumps({"target": "delinquency_rate", "frequency": "weekly"}),
        }

        tpl_path = f"{SQL_DIR}/mlops-model-registry.sql"
        if Path(tpl_path).exists():
            sql = Template(read_file(tpl_path)).render(**vals)
        else:
            # BigQuery MERGE (upsert)
            sql = f"""
            MERGE `{PROJECT}.{MLOPS_DATASET}.model` T
            USING (SELECT '{vals["model_id"]}' AS model_id) S
            ON T.model_id = S.model_id
            WHEN MATCHED THEN UPDATE SET
                model_name      = '{vals["model_name"]}',
                owner           = '{vals["owner"]}',
                business_problem= '{vals["business_problem"].replace("'", "''")}',
                ticket_number   = '{vals["ticket_number"]}',
                tags_json       = '{vals["tags_json"].replace("'", "''")}'
            WHEN NOT MATCHED THEN INSERT (model_id, model_name, owner, business_problem, ticket_number, tags_json, created_at)
            VALUES ('{vals["model_id"]}','{vals["model_name"]}','{vals["owner"]}',
                    '{vals["business_problem"].replace("'", "''")}','{vals["ticket_number"]}',
                    '{vals["tags_json"].replace("'", "''")}', CURRENT_TIMESTAMP());
            """
        run_execute(sql)
        print("✅ Model registered.")
        return vals

    @task
    def register_dataset():
        # row_count
        r = run_fetchone(f"""
            SELECT COUNT(*) AS c
            FROM `{PROJECT}.{GOLD_DATASET}.fact_all_indicators_weekly`
        """)
        row_count = int(r["c"]) if r and "c" in r.keys() else (int(r[0]) if r else 0)

        # feature_count (count json keys of one row)
        f = run_fetchone(f"""
            SELECT COUNT(*) AS f
            FROM UNNEST(REGEXP_EXTRACT_ALL(
              TO_JSON_STRING((SELECT AS STRUCT *
                              FROM `{PROJECT}.{GOLD_DATASET}.fact_all_indicators_weekly`
                              LIMIT 1)),
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
            VALUES ('{meta["dataset_id"]}','{meta["model_id"]}','{meta["data_version"]}',
                    {meta["row_count"]},{meta["feature_count"]}, CURRENT_TIMESTAMP())
            """
        run_execute(sql)
        print("✅ Dataset registered.")
        return meta

    # -----------------------------
    # Train and persist runs
    # -----------------------------
    @task(retries=0)
    def train_model(cfg: dict, ds_meta: dict):
        """POST to training CF with both querystring and JSON body."""
        run_id = f"run_{MODEL_ID}_{cfg['algorithm']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        hyperparams_str = json.dumps(cfg["hyperparameters"])
        metric_keys = ["smape", "rmse_recent6", "mae", "pearson_r", "r2", "mase"]
        metric_keys_str = json.dumps(metric_keys)

        params_qs = {
            "run_id": run_id,
            "model_id": MODEL_ID,
            "algorithm": cfg["algorithm"],
            "hyperparameters": hyperparams_str,   # JSON string
            "dataset_id": ds_meta["dataset_id"],
            "metric_keys": metric_keys_str,       # JSON string
        }
        payload_body = {
            "run_id": run_id,
            "model_id": MODEL_ID,
            "algorithm": cfg["algorithm"],
            "hyperparameters": cfg["hyperparameters"],  # dict
            "dataset_id": ds_meta["dataset_id"],
            "metric_keys": metric_keys,
        }

        print(f"🚀 Training {cfg['algorithm']} ...")
        resp = requests.post(
            CF_TRAIN_MODEL,
            params=params_qs,
            json=payload_body,
            headers={"Content-Type": "application/json"},
            timeout=300,
        )
        if resp.status_code >= 400:
            print(f"❌ Request failed ({resp.status_code}): {resp.text}")
            resp.raise_for_status()

        try:
            out = resp.json()
        except json.JSONDecodeError:
            out = {"text": resp.text}

        # normalize fields for downstream steps
        out.setdefault("run_id", run_id)
        out.setdefault("dataset_id", ds_meta["dataset_id"])
        out.setdefault("params", {
            "algorithm": cfg["algorithm"],
            "hyperparameters": cfg["hyperparameters"],
        })
        if "artifact" not in out and "gcs_path" in out:
            out["artifact"] = out["gcs_path"]

        return out

    @task(retries=0)
    def register_training_run(model_result: dict):
        sql = f"""
        INSERT INTO `{PROJECT}.{MLOPS_DATASET}.training_run`
        (run_id, model_id, dataset_id, params, metrics, artifact, status, created_at)
        VALUES (
            '{model_result["run_id"]}',
            '{MODEL_ID}',
            '{model_result.get("dataset_id","")}',
            '{json.dumps(model_result.get("params", {})).replace("'", "''")}',
            '{json.dumps(model_result.get("metrics", {})).replace("'", "''")}',
            '{model_result.get("artifact", model_result.get("gcs_path","")).replace("'", "''")}',
            '{model_result.get("status", "completed")}',
            CURRENT_TIMESTAMP()
        )
        """
        run_execute(sql)
        print(f"✅ Training run recorded: {model_result['run_id']}")
        return {"run_id": model_result["run_id"], "dataset_id": model_result.get("dataset_id", "")}

    # -----------------------------
    # Select best, version, deploy
    # -----------------------------
    @task
    def find_best_model(ds_meta: dict):
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
            raise AirflowSkipException("No completed training runs found.")

        base_sql = f"""
        SELECT JSON_VALUE(metrics, '$.smape') AS base_smape
        FROM `{PROJECT}.{MLOPS_DATASET}.training_run`
        WHERE model_id = '{MODEL_ID}'
          AND dataset_id = '{ds_meta["dataset_id"]}'
          AND status = 'completed'
          AND JSON_VALUE(params, '$.algorithm') = 'base'
        ORDER BY created_at DESC
        LIMIT 1
        """
        base = run_fetchone(base_sql)
        base_smape = None
        if base:
            try:
                base_smape = float(base["base_smape"]) if base["base_smape"] is not None else None
            except Exception:
                base_smape = float(base[0]) if base[0] is not None else None

        def _get(row, key, idx):
            try:
                return row[key]
            except Exception:
                return row[idx]

        return {
            "run_id": _get(best, "run_id", 0),
            "params": json.loads(_get(best, "params", 1)),
            "metrics": json.loads(_get(best, "metrics", 2)),
            "artifact": _get(best, "artifact", 3),
            "baseline_smape": base_smape,
            "dataset_id": ds_meta["dataset_id"],
        }

    @task
    def register_model_version(best: dict):
        status = "approved"
        if best.get("baseline_smape") is not None:
            new_smape = float(best["metrics"]["smape"])
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
        run_execute(insert_sql)
        print(f"✅ Model version recorded: {model_version_id} ({status})")
        return {"model_version_id": model_version_id, "status": status}

    @task
    def register_deployment(mv: dict):
        if mv["status"] != "approved":
            print("Deployment skipped (status != approved).")
            return {"deployed": False, "model_version_id": mv["model_version_id"]}

        # archive previous active deployments
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
        run_execute(archive_sql)

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
        run_execute(insert_sql)
        print(f"🚀 Deployed: {deployment_id} → {INFERENCE_ENDPOINT}")
        return {"deployed": True, "deployment_id": deployment_id, "endpoint_url": INFERENCE_ENDPOINT}

    # -----------------------------
    # Orchestration
    # -----------------------------
    # Ingest flows
    fred_extracts = extract_fred.expand(series_id=fred_series)
    fred_loads    = load_fred_to_bq.expand(payload=fred_extracts)
    fred_land     = load_fred_landing(fred_loads)

    yf_extracts   = extract_yfinance.expand(ticker=yfinance_tickers)
    yf_loads      = load_yfinance_to_bq.expand(payload=yf_extracts)
    yf_land       = load_yfinance_landing(yf_loads)

    # Build GOLD after both landings
    ml_ds = create_ml_dataset()
    ml_ds.set_upstream([fred_land, yf_land])

    # Register model & dataset snapshot
    model_reg   = register_model()
    dataset_reg = register_dataset()
    dataset_reg.set_upstream([ml_ds, model_reg])

    # Train models in parallel (fix: only train_model uses partial(ds_meta=...))
    train_results = train_model.partial(ds_meta=dataset_reg).expand(cfg=model_configs)

    # Persist runs (fix: NO partial here because function doesn't accept ds_meta/cfg)
    recorded_runs = register_training_run.expand(model_result=train_results)

    # Ensure best-model selection runs after all runs are recorded
    best = find_best_model(dataset_reg)
    best.set_upstream(recorded_runs)

    mv = register_model_version(best)
    register_deployment(mv)

# Instantiate DAG
credit_risk_pipeline()