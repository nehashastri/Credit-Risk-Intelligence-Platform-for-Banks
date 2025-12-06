import functions_framework
from google.cloud import bigquery, storage
import pandas as pd
import numpy as np
import json
import joblib
import io
from pathlib import Path

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
# Entry point
# -------------------------------------------------------------------------
@functions_framework.http
def task(request):
    """
    Serve predictions from the currently deployed credit delinquency model.
    
    Expected request params (all optional):
    - date: filter predictions by specific date (YYYY-MM-DD)
    - start_date: filter predictions from this date onwards
    - end_date: filter predictions up to this date
    - limit: limit number of results (default: 100)
    
    Returns predictions based on the deployed model type:
    - Python models: load .pkl from GCS and run predict()
    - Baseline models: simple constant prediction
    """
    
    # Get optional filter parameters
    date_filter = request.args.get("date")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    limit = request.args.get("limit", "100")
    
    try:
        limit = int(limit)
    except ValueError:
        limit = 100
    
    # Connect to BigQuery
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
        
        deployment_id = deployment_result.iloc[0]['deployment_id']
        model_version_id = deployment_result.iloc[0]['model_version_id']
        artifact_path = deployment_result.iloc[0]['artifact_path']
        metrics_json = deployment_result.iloc[0]['metrics_json']
        params_json = deployment_result.iloc[0]['params']
        
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
    
    # ---------------- Load feature data for prediction ----------------
    feature_query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{GOLD_DATASET}.fact_all_indicators_weekly`
        WHERE 1=1
    """
    
    # Add date filters if provided
    if date_filter:
        feature_query += f" AND DATE(date) = '{date_filter}'"
    elif start_date and end_date:
        feature_query += f" AND DATE(date) BETWEEN '{start_date}' AND '{end_date}'"
    elif start_date:
        feature_query += f" AND DATE(date) >= '{start_date}'"
    elif end_date:
        feature_query += f" AND DATE(date) <= '{end_date}'"
    
    # Order by date and limit
    feature_query += f" ORDER BY date DESC LIMIT {limit}"
    
    print(f"🔹 Loading feature data...")
    try:
        feature_df = bq.query(feature_query).to_dataframe()
        
        if len(feature_df) == 0:
            return {
                "error": "No data found",
                "details": "No data matching the provided filters"
            }, 404
        
        print(f"✅ Loaded {len(feature_df)} rows")
        
    except Exception as e:
        return {
            "error": "Error loading feature data",
            "details": str(e)
        }, 500
    
    # ---------------- Make predictions based on model type ----------------
    
    if algorithm == "base":
        # Baseline model - constant prediction
        print("🧠 Using baseline model...")
        try:
            # Load the baseline value from artifact (it's stored as a dict)
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
            
            # Create constant predictions
            predictions = np.full(len(feature_df), baseline_value)
            
        except Exception as e:
            print(f"⚠️ Could not load baseline model, using 0.0: {e}")
            predictions = np.zeros(len(feature_df))
    
    else:
        # Python/sklearn model - load from GCS
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
            
            # Prepare features (exclude target and date columns)
            feature_cols = [c for c in feature_df.columns 
                          if c not in ("delinq", "date", "week", "year")]
            
            X = feature_df[feature_cols].fillna(0)
            
            # Make predictions
            predictions = model.predict(X)
            print(f"✅ Generated {len(predictions)} predictions")
            
        except Exception as e:
            return {
                "error": "Error loading or using model",
                "details": str(e)
            }, 500
    
    # ---------------- Prepare response ----------------
    result_df = pd.DataFrame({
        'date': feature_df['date'],
        'week': feature_df.get('week', None),
        'year': feature_df.get('year', None),
        'predicted_delinquency_rate': predictions,
        'actual_delinquency_rate': feature_df.get('delinq', None)
    })
    
    # Calculate error if actuals are available
    if 'delinq' in feature_df.columns:
        result_df['error'] = result_df['predicted_delinquency_rate'] - result_df['actual_delinquency_rate']
        result_df['abs_error'] = result_df['error'].abs()
        result_df['pct_error'] = (result_df['error'] / result_df['actual_delinquency_rate'] * 100).round(2)
    
    # Parse metrics for display
    try:
        metrics = json.loads(metrics_json) if isinstance(metrics_json, str) else metrics_json
    except Exception:
        metrics = {}
    
    # Return as JSON
    response = {
        "deployment_id": deployment_id,
        "model_version_id": model_version_id,
        "algorithm": algorithm,
        "model_metrics": metrics,
        "predictions": result_df.to_dict(orient='records'),
        "count": len(result_df),
        "filters": {
            "date": date_filter,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit
        }
    }
    
    return response, 200