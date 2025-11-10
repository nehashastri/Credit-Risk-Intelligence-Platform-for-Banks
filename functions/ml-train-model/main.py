import functions_framework
from google.cloud import bigquery, storage
import pandas as pd
import json
import joblib
import io
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from jinja2 import Template

# Helper function to read SQL files
def read_sql(filename: str) -> str:
    """Read a SQL file from the sql/ directory."""
    base_dir = Path(__file__).parent

    # First, try /sql inside the function directory
    sql_path = base_dir / "include"/ "sql" / filename
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")
    return sql_path.read_text(encoding="utf-8")

# ---- SETTINGS ----
PROJECT_ID = "pipeline-882-team-project"
BUCKET_NAME = "pipeline-882-team-project-models"  # replace with your actual GCS bucket name
DATASET_TABLE = f"{PROJECT_ID}.gold.fact_all_indicators_weekly"

@functions_framework.http
def task(request):
    """
    Train a model using BigQuery data.
    
    Expected request params:
    - algorithm: "linear_regression" | "random_forest" | "gradient_boosting"
    - hyperparameters: JSON string of hyperparameters dict
    - run_id: training run identifier from Airflow
    - dataset_id: dataset identifier
    - model_id: model identifier
    """

    # Parse parameters from request
    algorithm = request.args.get("algorithm")
    hyperparams_json = request.args.get("hyperparameters", "{}")
    run_id = request.args.get("run_id")
    dataset_id = request.args.get("dataset_id")
    model_id = request.args.get("model_id")

    if not all([algorithm, run_id, dataset_id, model_id]):
        return {"error": "Missing required parameters"}, 400

    # Parse hyperparameters
    try:
        hyperparams = json.loads(hyperparams_json)
    except json.JSONDecodeError:
        return {"error": "Invalid hyperparameters JSON"}, 400

    # ---- CONNECT TO BIGQUERY ----
    bq = bigquery.Client(project=PROJECT_ID)

    # ---- LOAD TRAINING AND TEST DATA ----
    train_query = read_sql("load-train-data.sql")
    test_query = read_sql("load-test-data.sql")

    print("🔹 Loading training data from BigQuery...")
    train_df = bq.query(train_query).to_dataframe()

    print("🔹 Loading test data from BigQuery...")
    test_df = bq.query(test_query).to_dataframe()

    if "delinq" not in train_df.columns:
        return {"error": "Target variable 'delinq' not found in gold table."}, 400

    # ---- FEATURE SELECTION ----
    feature_cols = [col for col in train_df.columns 
                    if col not in ("delinq", "date", "week", "year")]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["delinq"].fillna(0)

    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["delinq"].fillna(0)

    # ---- MODEL TRAINING ----
    print(f"🧠 Training {algorithm} model...")

    if algorithm == "linear_regression":
        model = LinearRegression(**hyperparams)
    elif algorithm == "random_forest":
        model = RandomForestRegressor(**hyperparams, random_state=42)
    elif algorithm == "gradient_boosting":
        model = GradientBoostingRegressor(**hyperparams, random_state=42)
    else:
        return {"error": f"Unknown algorithm: {algorithm}"}, 400

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ---- EVALUATION ----
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else None

    print(f"✅ RMSE: {rmse:.4f}, MAE: {mae:.4f}, Corr: {corr:.4f}")

    # ---- SAVE MODEL ARTIFACT TO GCS ----
    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes)
    model_bytes.seek(0)

    gcs_path = f"models/model_id={model_id}/dataset_id={dataset_id}/run_id={run_id}/model.pkl"

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_file(model_bytes, content_type="application/octet-stream")

    gcs_full_path = f"gs://{BUCKET_NAME}/{gcs_path}"

    # ---- RETURN RESULTS ----
    result = {
        "run_id": run_id,
        "algorithm": algorithm,
        "gcs_path": gcs_full_path,
        "metrics": {
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
            "correlation": round(float(corr), 4) if corr else None,
            "test_count": len(test_df),
        },
        "feature_count": len(feature_cols),
    }

    print(json.dumps(result, indent=2))
    return result, 200
