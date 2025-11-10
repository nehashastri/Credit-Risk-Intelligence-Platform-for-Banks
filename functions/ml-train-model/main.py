import functions_framework
from google.cloud import bigquery, storage
import pandas as pd
import json
import joblib
import io
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from jinja2 import Template
import math

# Optional imports for XGBoost & LightGBM
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


# -------------------------------------------------------------------------
# Helper: Read SQL file
# -------------------------------------------------------------------------
def read_sql(filename: str) -> str:
    """Read a SQL file from the sql/ directory."""
    base_dir = Path(__file__).parent
    possible_paths = [base_dir / filename, base_dir / "sql" / filename]

    for path in possible_paths:
        if path.exists():
            print(f"✅ Found SQL file: {path}")
            return path.read_text(encoding="utf-8")

    raise FileNotFoundError(f"SQL file not found: {filename}")


# -------------------------------------------------------------------------
# Helper metrics
# -------------------------------------------------------------------------
def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.maximum(denom, 1e-8)
    return 100 * np.mean(diff)


def rmse(y_true, y_pred):
    return math.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def mase(y_true, y_pred):
    """Mean Absolute Scaled Error — scaled to lag-1 naïve forecast."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    return mae_model / (mae_naive + 1e-8)


# -------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------
PROJECT_ID = "pipeline-882-team-project"
BUCKET_NAME = "pipeline-882-team-project-models"
DATASET_TABLE = f"{PROJECT_ID}.gold.fact_all_indicators_weekly"


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
@functions_framework.http
def task(request):
    """
    Train and evaluate model with extended metrics:
    sMAPE, RMSE_recent6, MAE, Pearson r, R², MASE
    """

    # ---------------- Parse params ----------------
    algorithm = request.args.get("algorithm")
    hyperparams_json = request.args.get("hyperparameters", "{}")
    run_id = request.args.get("run_id")
    dataset_id = request.args.get("dataset_id")
    model_id = request.args.get("model_id")

    if not all([algorithm, run_id, dataset_id, model_id]):
        return {"error": "Missing required parameters"}, 400

    try:
        hyperparams = json.loads(hyperparams_json)
    except json.JSONDecodeError:
        return {"error": "Invalid hyperparameters JSON"}, 400

    # ---------------- Load data ----------------
    bq = bigquery.Client(project=PROJECT_ID)
    train_query = read_sql("load-train-data.sql")
    test_query = read_sql("load-test-data.sql")

    print("🔹 Loading training data...")
    train_df = bq.query(train_query).to_dataframe()

    print("🔹 Loading test data...")
    test_df = bq.query(test_query).to_dataframe()

    if "delinq" not in train_df.columns:
        return {"error": "Target variable 'delinq' not found."}, 400

    # ---------------- Feature prep ----------------
    feature_cols = [c for c in train_df.columns if c not in ("delinq", "date", "week", "year")]
    X_train, y_train = train_df[feature_cols].fillna(0), train_df["delinq"].fillna(0)
    X_test, y_test = test_df[feature_cols].fillna(0), test_df["delinq"].fillna(0)

    # ---------------- Model training ----------------
    print(f"🧠 Training {algorithm}...")
    if algorithm == "base":
        baseline = pd.Series(y_train).rolling(window=3, min_periods=1).mean().iloc[-1]
        y_pred = np.full_like(y_test, baseline, dtype=float)
        model = {"type": "baseline", "value": baseline}

    elif algorithm == "linear_regression":
        model = LinearRegression(**hyperparams)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif algorithm == "elastic_net":
        model = ElasticNet(**hyperparams, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif algorithm == "random_forest":
        model = RandomForestRegressor(**hyperparams, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif algorithm == "gradient_boosting":
        model = GradientBoostingRegressor(**hyperparams, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif algorithm == "xgboost":
        if xgb is None:
            return {"error": "xgboost not installed"}, 400
        model = xgb.XGBRegressor(**hyperparams, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif algorithm == "lightgbm":
        if lgb is None:
            return {"error": "lightgbm not installed"}, 400
        model = lgb.LGBMRegressor(**hyperparams, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    else:
        return {"error": f"Unknown algorithm: {algorithm}"}, 400

    # ---------------- Metrics ----------------
    pearson_r = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else None
    smape_val = smape(y_test, y_pred)
    mae_val = mean_absolute_error(y_test, y_pred)
    mase_val = mase(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)
    rmse_recent6_val = rmse(y_test[-6:], y_pred[-6:]) if len(y_test) >= 6 else rmse(y_test, y_pred)

    print(f"✅ Pearson r: {pearson_r:.4f}" if pearson_r else "⚠️ Not enough data for correlation")
    print(f"✅ sMAPE: {smape_val:.2f}%")
    print(f"✅ MAE: {mae_val:.4f}")
    print(f"✅ MASE: {mase_val:.4f}")
    print(f"✅ R²: {r2_val:.4f}")
    print(f"✅ RMSE_recent6: {rmse_recent6_val:.4f}")

    # ---------------- Save model ----------------
    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes)
    model_bytes.seek(0)

    gcs_path = f"models/model_id={model_id}/dataset_id={dataset_id}/run_id={run_id}/model.pkl"
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    bucket.blob(gcs_path).upload_from_file(model_bytes, content_type="application/octet-stream")

    # ---------------- Return response ----------------
    result = {
        "run_id": run_id,
        "algorithm": algorithm,
        "gcs_path": f"gs://{BUCKET_NAME}/{gcs_path}",
        "metrics": {
            "pearson_r": round(float(pearson_r), 4) if pearson_r else None,
            "smape": round(float(smape_val), 2),
            "mae": round(float(mae_val), 4),
            "mase": round(float(mase_val), 4),
            "r2": round(float(r2_val), 4),
            "rmse_recent6": round(float(rmse_recent6_val), 4),
            "test_count": len(y_test)
        },
        "feature_count": len(feature_cols),
    }

    print(json.dumps(result, indent=2))
    return result, 200
