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
from jinja2 import Template


# -------------------------------------------------------------------------
# Helper function to read SQL files
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Helper function to read SQL files (with debugging)
# -------------------------------------------------------------------------
def read_sql(filename: str) -> str:
    """
    Read a SQL file from the sql/ directory. Includes debug info
    to inspect deployed directory structure on Cloud Functions.
    """
    import os

    base_dir = Path(__file__).parent
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📂 __file__ location: {__file__}")
    print(f"📂 Base dir: {base_dir}")

    # Try multiple possible locations
    possible_paths = [
        base_dir / filename,
        base_dir / "sql" / filename
    ]

    for path in possible_paths:
        print(f"🔍 Checking for: {path}")
        if path.exists():
            print(f"✅ Found SQL file at: {path}")
            return path.read_text(encoding="utf-8")

    # If not found
    print("❌ None of the expected paths contain the SQL file.")
    raise FileNotFoundError(f"SQL file not found in any known path for {filename}")


# -------------------------------------------------------------------------
# Helper: Symmetric Mean Absolute Percentage Error
# -------------------------------------------------------------------------
def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8)
    return 100 * np.mean(diff)


# -------------------------------------------------------------------------
# Project Settings
# -------------------------------------------------------------------------
PROJECT_ID = "pipeline-882-team-project"
BUCKET_NAME = "pipeline-882-team-project-models"
DATASET_TABLE = f"{PROJECT_ID}.gold.fact_all_indicators_weekly"


# -------------------------------------------------------------------------
# Cloud Function Entry Point
# -------------------------------------------------------------------------
@functions_framework.http
def task(request):
    """
    Train a model using BigQuery data.

    Expected query params:
      - algorithm: "linear_regression" | "random_forest" | "gradient_boosting"
      - hyperparameters: JSON string
      - run_id: training run identifier
      - dataset_id: dataset identifier
      - model_id: model identifier
    """

    # ---------------------------------------------------------------------
    # Parse parameters
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Load Data from BigQuery
    # ---------------------------------------------------------------------
    bq = bigquery.Client(project=PROJECT_ID)

    train_query = read_sql("load-train-data.sql")
    test_query = read_sql("load-test-data.sql")

    print("🔹 Loading training data from BigQuery...")
    train_df = bq.query(train_query).to_dataframe()

    print("🔹 Loading test data from BigQuery...")
    test_df = bq.query(test_query).to_dataframe()

    if "delinq" not in train_df.columns:
        return {"error": "Target variable 'delinq' not found in gold table."}, 400

    # ---------------------------------------------------------------------
    # Feature Engineering
    # ---------------------------------------------------------------------
    feature_cols = [
        col for col in train_df.columns
        if col not in ("delinq", "date", "week", "year")
    ]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["delinq"].fillna(0)

    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["delinq"].fillna(0)

    # ---------------------------------------------------------------------
    # Model Training
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Evaluation: Pearson r and SMAPE
    # ---------------------------------------------------------------------
    if len(y_test) > 1:
        pearson_r = np.corrcoef(y_test, y_pred)[0, 1]
    else:
        pearson_r = None

    smape_val = smape(y_test, y_pred)

    print(f"✅ Pearson r: {pearson_r:.4f}" if pearson_r is not None else "⚠️ Not enough data for correlation")
    print(f"✅ SMAPE: {smape_val:.2f}%")

    # ---------------------------------------------------------------------
    # Save Model Artifact to GCS
    # ---------------------------------------------------------------------
    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes)
    model_bytes.seek(0)

    gcs_path = (
        f"models/model_id={model_id}/dataset_id={dataset_id}/run_id={run_id}/model.pkl"
    )

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_file(model_bytes, content_type="application/octet-stream")

    gcs_full_path = f"gs://{BUCKET_NAME}/{gcs_path}"

    # ---------------------------------------------------------------------
    # Return Results
    # ---------------------------------------------------------------------
    result = {
        "run_id": run_id,
        "algorithm": algorithm,
        "gcs_path": gcs_full_path,
        "metrics": {
            "pearson_r": round(float(pearson_r), 4) if pearson_r else None,
            "smape": round(float(smape_val), 2),
            "test_count": len(test_df),
        },
        "feature_count": len(feature_cols),
    }

    print(json.dumps(result, indent=2))
    return result, 200
