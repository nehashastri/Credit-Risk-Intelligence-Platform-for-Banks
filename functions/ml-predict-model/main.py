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
# Settings
# -------------------------------------------------------------------------
PROJECT_ID = "pipeline-882-team-project"
BUCKET_NAME = "pipeline-882-team-project-models"
MLOPS_DATASET = "mlops"
GOLD_DATASET = "gold"


# -------------------------------------------------------------------------
# Helper: Read SQL file (optional, if you have SQL files)
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
# Helper: convert ISO date string → (iso_year, iso_week)
# -------------------------------------------------------------------------
def date_to_year_week(date_str: str) -> tuple[int, int]:
    """
    Convert a YYYY-MM-DD string into (ISO year, ISO week).
    Only used for the historical GOLD table.
    """
    d = datetime.fromisoformat(date_str)
    iso_year, iso_week, _ = d.isocalendar()
    return int(iso_year), int(iso_week)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
@functions_framework.http
def task(request):
    """
    Serve predictions from the currently deployed credit delinquency model.

    Expected request params (all optional, query string):

    - source: which feature table to use
        * "gold" (default) → gold.fact_all_indicators_weekly  (historical)
        * "scenario_mean"  → gold.fact_all_indicators_weekly_llm_scenario_mean
                             (LLM future scenarios)

    - date:       filter by a specific date (YYYY-MM-DD)
    - start_date: filter from this date onwards (YYYY-MM-DD)
    - end_date:   filter up to this date (YYYY-MM-DD)
    - limit:      maximum number of rows to return (default: 100)

    Returns predictions based on the deployed model type:
    - Python models: load .pkl from GCS and run predict()
    - Baseline models: simple constant prediction
    """

    # ---------------- Read query params ----------------
    source = request.args.get("source", "gold")  # "gold" or "scenario_mean"

    date_filter_str = request.args.get("date")
    start_date_str = request.args.get("start_date")
    end_date_str = request.args.get("end_date")

    limit = request.args.get("limit", "100")
    try:
        limit = int(limit)
    except ValueError:
        limit = 100

    # ---------------- BigQuery client ----------------
    bq = bigquery.Client(project=PROJECT_ID)

    # ---------------- Look up current deployment ----------------
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
                "details": "No model has been deployed yet. Run the training pipeline first."
            }, 404

        deployment_id = deployment_result.iloc[0]["deployment_id"]
        model_version_id = deployment_result.iloc[0]["model_version_id"]
        artifact_path = deployment_result.iloc[0]["artifact_path"]
        metrics_json = deployment_result.iloc[0]["metrics_json"]
        params_json = deployment_result.iloc[0]["params"]

        # Parse params to get algorithm
        params = json.loads(params_json) if isinstance(params_json, str) else params_json
        algorithm = params.get("algorithm", "unknown")

        print(f"✅ Using deployment: {deployment_id}")
        print(f"✅ Model version: {model_version_id}")
        print(f"✅ Algorithm: {algorithm}")
        print(f"✅ Artifact path: {artifact_path}")

    except Exception as e:
        return {
            "error": "Could not lookup deployment",
            "details": str(e)
        }, 500

    # ---------------- Decide which feature table to query ----------------
    if source == "scenario_mean":
        feature_table = (
            f"`{PROJECT_ID}.{GOLD_DATASET}.fact_all_indicators_weekly_llm_scenario_mean`"
        )
    else:
        feature_table = f"`{PROJECT_ID}.{GOLD_DATASET}.fact_all_indicators_weekly`"

    print(f"🔹 Using feature source: {source} ({feature_table})")

    # ---------------- Build feature query ----------------
    # NOTE:
    # - For historical "gold" table, we filter by (year, week)
    # - For "scenario_mean" table, we filter directly by DATE(date)
    if source == "scenario_mean":
        feature_query = f"""
            SELECT *
            FROM {feature_table}
            WHERE 1=1
        """

        # Use DATE(date) filters
        if date_filter_str:
            feature_query += f" AND DATE(date) = DATE('{date_filter_str}')"
        elif start_date_str and end_date_str:
            feature_query += (
                f" AND DATE(date) BETWEEN DATE('{start_date_str}') "
                f"AND DATE('{end_date_str}')"
            )
        elif start_date_str:
            feature_query += f" AND DATE(date) >= DATE('{start_date_str}')"
        elif end_date_str:
            feature_query += f" AND DATE(date) <= DATE('{end_date_str}')"

        feature_query += f" ORDER BY date ASC LIMIT {limit}"

    else:
        # Historical GOLD table → use (year, week) logic
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
                    AND (
                        year > {sy} OR (year = {sy} AND week >= {sw})
                    )
                """
            elif end_date_str:
                ey, ew = date_to_year_week(end_date_str)
                feature_query += f"""
                    AND (
                        year < {ey} OR (year = {ey} AND week <= {ew})
                    )
                """
        except Exception as e:
            # If parsing fails, just log it and continue without date filter
            print(f"⚠️ Failed to parse date filters: {e}")

        feature_query += f" ORDER BY year DESC, week DESC LIMIT {limit}"

    print("🔹 Feature query:")
    print(feature_query)

    # ---------------- Load feature data for prediction ----------------
    try:
        feature_df = bq.query(feature_query).to_dataframe()

        if len(feature_df) == 0:
            return {
                "error": "No data found",
                "details": "No data matching the provided filters"
            }, 404

        print(f"✅ Loaded {len(feature_df)} rows from {source}")

    except Exception as e:
        return {
            "error": "Error loading feature data",
            "details": str(e)
        }, 500

    # ---------------- Ensure 'rn' column exists for the model ----------------
    if "rn" not in feature_df.columns:
        feature_df["rn"] = np.arange(len(feature_df))
        print("ℹ️ Column 'rn' not found in features. Created sequential rn 0..N-1.")

    # ---------------- Make predictions based on model type ----------------
    if algorithm == "base":
        # Baseline model - constant prediction
        print("🧠 Using baseline model...")
        try:
            # Load the baseline value from artifact (stored as a dict)
            storage_client = storage.Client()
            path_parts = artifact_path.replace("gs://", "").split("/", 1)
            gcs_bucket = path_parts[0]
            gcs_blob_path = path_parts[1]

            bucket = storage_client.bucket(gcs_bucket)
            blob = bucket.blob(gcs_blob_path)

            model_bytes = io.BytesIO()
            blob.download_to_file(model_bytes)
            model_bytes.seek(0)

            baseline_model = joblib.load(model_bytes)
            baseline_value = baseline_model.get("value", 0.0)

            predictions = np.full(len(feature_df), baseline_value)

        except Exception as e:
            print(f"⚠️ Could not load baseline model, using 0.0: {e}")
            predictions = np.zeros(len(feature_df))

    else:
        # Python / sklearn model - load from GCS
        print(f"🧠 Using {algorithm} model - loading from GCS...")
        try:
            # Parse GCS path: gs://bucket/path/to/model.pkl
            path_parts = artifact_path.replace("gs://", "").split("/", 1)
            if len(path_parts) != 2:
                return {"error": "Invalid GCS path format"}, 400

            gcs_bucket = path_parts[0]
            gcs_blob_path = path_parts[1]

            # Load model from GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(gcs_bucket)
            blob = bucket.blob(gcs_blob_path)

            model_bytes = io.BytesIO()
            blob.download_to_file(model_bytes)
            model_bytes.seek(0)

            model = joblib.load(model_bytes)
            print("✅ Model loaded successfully")

            # Prepare features (exclude target and key columns)
            feature_cols = [
                c
                for c in feature_df.columns
                if c not in ("delinq", "date", "week", "year")
            ]

            X = feature_df[feature_cols].fillna(0)

            predictions = model.predict(X)
            print(f"✅ Generated {len(predictions)} predictions")

        except Exception as e:
            return {
                "error": "Error loading or using model",
                "details": str(e)
            }, 500

    # ---------------- Prepare response ----------------
    # Build a date-like label for plotting
    if source == "scenario_mean":
        # Use actual date column from the LLM scenario table
        if "date" in feature_df.columns:
            date_str = pd.to_datetime(feature_df["date"]).dt.strftime("%Y-%m-%d")
        else:
            # Fallback: horizon index if date is missing
            date_str = [f"H{idx+1}" for idx in range(len(feature_df))]
    else:
        # For historical GOLD table, build "YYYY-Www" label
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

    # Add error metrics only for historical data
    if "delinq" in feature_df.columns and source == "gold":
        result_df["error"] = (
            result_df["predicted_delinquency_rate"]
            - result_df["actual_delinquency_rate"]
        )
        result_df["abs_error"] = result_df["error"].abs()
        result_df["pct_error"] = (
            result_df["error"] / result_df["actual_delinquency_rate"] * 100
        ).round(2)

    # Parse metrics for display
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