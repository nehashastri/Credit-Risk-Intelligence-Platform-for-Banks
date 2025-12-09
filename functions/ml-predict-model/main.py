import functions_framework
from google.cloud import bigquery, storage
import pandas as pd
import numpy as np
import json
import joblib
import io
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------------------------
# Global Settings
# -------------------------------------------------------------------------
PROJECT_ID = "pipeline-882-team-project"
BUCKET_NAME = "pipeline-882-team-project-models"
MLOPS_DATASET = "mlops"
GOLD_DATASET = "gold"

# -------------------------------------------------------------------------
# Helper: Read SQL file
# -------------------------------------------------------------------------
def read_sql(filename: str) -> str:
    """
    Try to read a SQL file from the local filesystem.
    Supports both root and a "sql" subdirectory.
    """
    base_dir = Path(__file__).parent
    possible_paths = [base_dir / filename, base_dir / "sql" / filename]

    for path in possible_paths:
        if path.exists():
            print(f"Found SQL file: {path}")
            return path.read_text(encoding="utf-8")

    raise FileNotFoundError(f"SQL file not found: {filename}")

# -------------------------------------------------------------------------
# Helper: Convert YYYY-MM-DD -> (ISO year, ISO week)
# -------------------------------------------------------------------------
def date_to_year_week(date_str: str) -> tuple[int, int]:
    """
    Convert an ISO date string (YYYY-MM-DD) into (ISO year, ISO week).
    Used to filter feature tables by (year, week).
    """
    d = datetime.fromisoformat(date_str)
    iso_year, iso_week, _ = d.isocalendar()
    return int(iso_year), int(iso_week)

# -------------------------------------------------------------------------
# Main HTTP Cloud Function (Prediction API)
# -------------------------------------------------------------------------
@functions_framework.http
def task(request):
    """
    Serve predictions from the currently deployed credit delinquency model.

    Supported query parameters:
      - source: "gold" (default) or "scenario_mean"
      - date: ISO date string YYYY-MM-DD
      - start_date, end_date: date range
      - limit: max rows (default=100)

    Logic:
      1) Identify the current model deployment (Mlops.deployment table)
      2) Select the feature table depending on 'source'
      3) Filter the feature table by date/year/week
      4) Load model artifact from GCS
      5) Predict delinquency
      6) Return JSON result
    """

    # Read query parameters
    source = request.args.get("source", "gold")

    date_filter_str = request.args.get("date")
    start_date_str = request.args.get("start_date")
    end_date_str = request.args.get("end_date")

    limit = request.args.get("limit", "100")
    try:
        limit = int(limit)
    except ValueError:
        limit = 100

    # BigQuery client
    bq = bigquery.Client(project=PROJECT_ID)

    # Query the most recent model with >0 traffic_split
    deployment_query = f"""
        SELECT 
            d.deployment_id,
            d.model_version_id,
            d.endpoint_url,
            d.traffic_split,
            mv.artifact_path,
            mv.metrics_json,
            tr.params
        FROM `{PROJECT_ID}.{MLOPS_DATASET}.deployment` d
        JOIN `{PROJECT_ID}.{MLOPS_DATASET}.model_version` mv 
            ON d.model_version_id = mv.model_version_id
        JOIN `{PROJECT_ID}.{MLOPS_DATASET}.training_run` tr 
            ON mv.training_run_id = tr.run_id
        WHERE d.traffic_split > 0
        AND mv.model_id = 'credit_delinquency_model'
        ORDER BY d.deployed_at DESC
        LIMIT 1
    """

    try:
        deployment_result = bq.query(deployment_query).to_dataframe()
        if len(deployment_result) == 0:
            return {
                "error": "No deployment found",
                "details": "No model has been deployed yet"
            }, 404

        deployment_id = deployment_result.iloc[0]["deployment_id"]
        model_version_id = deployment_result.iloc[0]["model_version_id"]
        artifact_path = deployment_result.iloc[0]["artifact_path"]
        metrics_json = deployment_result.iloc[0]["metrics_json"]
        params_json = deployment_result.iloc[0]["params"]

        params = json.loads(params_json) if isinstance(params_json, str) else params_json
        algorithm = params.get("algorithm", "unknown")

    except Exception as e:
        return {"error": "Could not lookup deployment", "details": str(e)}, 500

    # Select the feature table
    if source == "scenario_mean":
        feature_table = (
            f"`{PROJECT_ID}.{GOLD_DATASET}.fact_all_indicators_weekly_llm_scenario_mean`"
        )
    else:
        feature_table = f"`{PROJECT_ID}.{GOLD_DATASET}.fact_all_indicators_weekly`"

    # Build feature query
    feature_query = f"""
        SELECT *
        FROM {feature_table}
        WHERE 1=1
    """

    try:
        if date_filter_str:
            y, w = date_to_year_week(date_filter_str)
            feature_query += f" AND year = {y} AND week = {w}"

        elif start_date_str and end_date_str:
            sy, sw = date_to_year_week(start_date_str)
            ey, ew = date_to_year_week(end_date_str)
            feature_query += f"""
                AND (
                    (year > {sy} OR (year = {sy} AND week >= {sw}))
                    AND
                    (year < {ey} OR (year = {ey} AND week <= {ew}))
                )
            """

        elif start_date_str:
            sy, sw = date_to_year_week(start_date_str)
            feature_query += f"""
                AND (year > {sy} OR (year = {sy} AND week >= {sw}))
            """

        elif end_date_str:
            ey, ew = date_to_year_week(end_date_str)
            feature_query += f"""
                AND (year < {ey} OR (year = {ey} AND week <= {ew}))
            """
    except Exception as e:
        print(f"Failed to parse date filters: {e}")

    if source == "scenario_mean":
        feature_query += f" ORDER BY year ASC, week ASC LIMIT {limit}"
    else:
        feature_query += f" ORDER BY year DESC, week DESC LIMIT {limit}"

    # Load feature rows
    try:
        feature_df = bq.query(feature_query).to_dataframe()

        if len(feature_df) == 0:
            return {
                "error": "No data found",
                "details": "No feature rows matching filters"
            }, 404

    except Exception as e:
        return {
            "error": "Error loading feature data",
            "details": str(e)
        }, 500

    # Ensure `rn` exists (model expects this column)
    if "rn" not in feature_df.columns:
        feature_df["rn"] = np.arange(len(feature_df))

    # Prediction logic
    if algorithm == "base":
        # Baseline model (constant prediction)
        try:
            storage_client = storage.Client()
            parts = artifact_path.replace("gs://", "").split("/", 1)
            bucket = storage_client.bucket(parts[0])
            blob = bucket.blob(parts[1])

            buf = io.BytesIO()
            blob.download_to_file(buf)
            buf.seek(0)

            baseline_model = joblib.load(buf)
            baseline_value = baseline_model.get("value", 0.0)
            predictions = np.full(len(feature_df), baseline_value)

        except Exception:
            predictions = np.zeros(len(feature_df))

    else:
        # Sklearn model
        try:
            parts = artifact_path.replace("gs://", "").split("/", 1)
            storage_client = storage.Client()
            bucket = storage_client.bucket(parts[0])
            blob = bucket.blob(parts[1])

            buf = io.BytesIO()
            blob.download_to_file(buf)
            buf.seek(0)

            model = joblib.load(buf)

            feature_cols = [
                c
                for c in feature_df.columns
                if c not in ("delinq", "date", "week", "year")
            ]
            X = feature_df[feature_cols].fillna(0)
            predictions = model.predict(X)

        except Exception as e:
            return {"error": "Model load error", "details": str(e)}, 500

    # Prepare response dataframe
    if source == "scenario_mean":
        date_str = [f"H{idx+1}" for idx in range(len(feature_df))]
    else:
        date_str = (
            feature_df["year"].astype(str)
            + "-W"
            + feature_df["week"].astype(str).str.zfill(2)
        )

    result_df = pd.DataFrame(
        {
            "date": date_str,
            "week": feature_df.get("week", None),
            "year": feature_df.get("year", None),
            "predicted_delinquency_rate": predictions,
            "actual_delinquency_rate": feature_df.get("delinq", None),
        }
    )

    if "delinq" in feature_df.columns and source == "gold":
        result_df["error"] = (
            result_df["predicted_delinquency_rate"]
            - result_df["actual_delinquency_rate"]
        )
        result_df["abs_error"] = result_df["error"].abs()
        result_df["pct_error"] = (
            result_df["error"] / result_df["actual_delinquency_rate"] * 100
        ).round(2)

    # Parse metrics
    try:
        metrics = json.loads(metrics_json) if isinstance(metrics_json, str) else metrics_json
    except Exception:
        metrics = {}

    response = {
        "deployment_id": deployment_id,
        "model_version_id": model_version_id,
        "algorithm": algorithm,
        "model_metrics": metrics,
        "source": source,
        "predictions": result_df.to_dict(orient="records"),
        "count": len(result_df),
        "filters": {
            "date": date_filter_str,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "limit": limit,
        },
    }

    return response, 200