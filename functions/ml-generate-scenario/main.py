import os
import io
import json
import math
from datetime import datetime

import functions_framework
import pandas as pd
import numpy as np

from google.cloud import bigquery, storage
from openai import OpenAI
import re

# =============================================================================
# Global Configuration
# =============================================================================
# Project ID for BigQuery and GCS access
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "pipeline-882-team-project")

# Mlops BigQuery dataset (stores model deployments, versions, training runs)
MLOPS_DATASET = "mlops"

# GOLD BigQuery dataset (stores historical aggregate features)
GOLD_DATASET = "gold"

# OpenAI API key is expected to be provided by GCP Secret Manager
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Default LLM model name (can be overridden via environment variable)
OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-mini")

# Default bucket storing model artifacts (joblib serialized models)
DEFAULT_BUCKET = os.getenv("MODEL_BUCKET", "pipeline-882-team-project-models")


# =============================================================================
# Utility Helpers
# =============================================================================
def safe_to_float(x):
    """
    Safely convert a value to float. Return None if conversion fails
    or results in NaN/inf. Used for numeric cleanup.
    """
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def get_bq():
    """
    Create a new BigQuery client using the configured project ID.
    """
    return bigquery.Client(project=PROJECT_ID)


def get_openai_client():
    """
    Create an OpenAI client instance using the API key loaded from environment.
    This will raise an error if the environment variable is not set.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=OPENAI_API_KEY)


def load_model_from_gcs(artifact_path: str):
    """
    Load a trained model object from a GCS URI.
    Expected artifact_path format: gs://bucket/path/to/model.pkl
    """
    from joblib import load as joblib_load

    if not artifact_path:
        raise RuntimeError("artifact_path is empty.")

    storage_client = storage.Client()
    path = artifact_path.replace("gs://", "")
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise RuntimeError(f"Invalid artifact path: {artifact_path}")

    bucket_name, blob_path = parts
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)

    model = joblib_load(buf)
    return model


def get_current_deployment():
    """
    Query BigQuery MLOps dataset to retrieve the most recent active deployment
    (traffic_split > 0) for the model_id='credit_delinquency_model'.

    Returns a dictionary containing:
      - deployment_id
      - model_version_id
      - artifact_path
      - metrics_json
      - algorithm name
    """
    bq = get_bq()
    query = f"""
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
    df = bq.query(query).to_dataframe()
    if df.empty:
        raise RuntimeError("No active model deployment found.")

    row = df.iloc[0]
    params_json = row["params"]
    try:
        params = json.loads(params_json) if isinstance(params_json, str) else params_json
    except Exception:
        params = {}

    return {
        "deployment_id": row["deployment_id"],
        "model_version_id": row["model_version_id"],
        "artifact_path": row["artifact_path"],
        "metrics_json": row["metrics_json"],
        "algorithm": params.get("algorithm", "unknown"),
    }


# =============================================================================
# Historical Feature Loading
# =============================================================================
def load_hist_and_features():
    """
    Load the most recent 24 weeks from the GOLD historical feature table:
        gold.fact_all_indicators_weekly

    Steps:
      1. Query the last 24 rows sorted by (year DESC, week DESC)
      2. Re-sort them back to ascending chronological order
      3. Extract the feature column list (excluding date/year/week/target)
      4. Ensure 'rn' exists as a stable index for model input

    Returns:
      - hist_df: DataFrame of 24 historical rows
      - feature_cols: list of model feature column names
    """
    bq = get_bq()

    sql_hist = f"""
        SELECT *
        FROM `{PROJECT_ID}.{GOLD_DATASET}.fact_all_indicators_weekly`
        ORDER BY year DESC, week DESC
        LIMIT 24
    """
    hist_df = bq.query(sql_hist).to_dataframe()
    if hist_df.empty:
        raise RuntimeError("No historical data found.")

    hist_df = hist_df.sort_values(["year", "week"]).reset_index(drop=True)

    if "rn" not in hist_df.columns:
        hist_df["rn"] = np.arange(len(hist_df))

    exclude_cols = {
        "date",
        "year",
        "week",
        "delinq",
        "target",
        "credit_delinquency_rate",
    }
    feature_cols = [c for c in hist_df.columns if c not in exclude_cols]

    if "rn" not in feature_cols:
        feature_cols.append("rn")

    return hist_df, feature_cols


def load_recent_news(limit=30):
    """
    Load the most recent economic news articles from:
        landing.news_articles

    Returns a structured list of items containing:
      published_at, title, topic, score, snippet
    """
    bq = get_bq()
    sql = f"""
        SELECT
          published_at,
          title,
          topic,
          score,
          snippet
        FROM `{PROJECT_ID}.landing.news_articles`
        WHERE published_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        ORDER BY published_at DESC
        LIMIT {limit}
    """
    rows = list(bq.query(sql).result())
    items = []
    for r in rows:
        items.append(
            {
                "published_at": str(r["published_at"]),
                "title": r["title"],
                "topic": r.get("topic") or "",
                "score": r.get("score"),
                "snippet": r.get("snippet") or "",
            }
        )
    return items


# =============================================================================
# LLM Scenario Generation
# =============================================================================
def strip_markdown_code_fence(text: str) -> str:
    """
    Remove possible markdown code fences around JSON output:
        ```json
        { ... }
        ```
    This function extracts only the JSON body.
    """
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"```$", "", t).strip()
    return t


def call_llm_for_scenario(
    feature_cols,
    hist_sample_df,
    news_items,
    scenario_text: str,
    horizon_weeks: int = 8,
):
    """
    Use the OpenAI completion API to generate a plausible forward scenario
    for macroeconomic features over N weeks.

    Prompt Includes:
      - Historical feature rows (last 8 weeks)
      - Recent economic news (optional context)
      - User-provided scenario text (e.g., change in unemployment rate)
      - Weekly feature schema

    Expected Output Format:
    {
      "rows": [
        {
          "date": "YYYY-MM-DD",
          "<feature1>": <numeric>,
          ...
        },
        ...
      ]
    }

    The function:
      1. Builds a prompt with system/user roles
      2. Calls OpenAI Chat Completions
      3. Parses JSON output
      4. Returns scenario rows
    """
    client = get_openai_client()

    hist_sample = hist_sample_df.tail(8).to_dict(orient="records")

    news_text = "\n\n".join(
        f"[{n['published_at']}] {n['title']} (topic={n['topic']}, score={n['score']})\n{n['snippet']}"
        for n in news_items
    )

    system_prompt = """
You are a macroeconomic scenario generator for a credit risk model.
Your role is to generate plausible weekly values for the future.

Return ONLY a single JSON object, with NO explanation, NO markdown, NO code fences.

Format:
{
  "rows": [
    {
      "date": "YYYY-MM-DD",
      "<feature1>": 0.0,
      ...
    },
    ...
  ]
}

Rules:
- Generate exactly N weekly rows.
- "date" must be a valid ISO date.
- Include ALL feature columns.
- Use ONLY numeric values.
- No target/delinquency.
"""

    user_prompt = f"""
Weekly schema:
- date
- Features: {feature_cols}

Recent historical data:
{json.dumps(hist_sample, indent=2)}

Recent news:
{news_text}

User scenario:
\"\"\"{scenario_text}\"\"\"

Requirements:
- Generate {horizon_weeks} future weekly rows.
- Begin from the next week after the most recent historical date.
- Include ALL feature columns listed above.
- Return ONLY the JSON object described.
"""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.3,
    )

    text = resp.choices[0].message.content
    text = strip_markdown_code_fence(text)

    try:
        data = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Failed to parse LLM JSON: {e} | text={text[:500]}")

    rows = data.get("rows") or []
    if not rows:
        raise RuntimeError("LLM returned empty 'rows'.")

    return rows


# =============================================================================
# HTTP Cloud Function Entry Point
# =============================================================================
@functions_framework.http
def task(request):
    """
    HTTP Cloud Function for scenario simulation + prediction.

    Expected JSON body:
    {
      "scenario_text": "If unemployment rises by 1 pp...",
      "horizon_weeks": 8
    }

    Steps:
      1. Load historical data for feature context
      2. Load recent economic news
      3. Use LLM to generate forward macro scenario
      4. Load deployed model artifact
      5. Predict delinquency on generated scenario
      6. Return JSON response
    """
    try:
        try:
            body = request.get_json(silent=True) or {}
        except Exception:
            body = {}

        scenario_text = body.get("scenario_text") or body.get("query") or ""
        horizon_weeks = int(body.get("horizon_weeks", 8))

        if not scenario_text:
            err = {"error": "scenario_text is required."}
            return json.dumps(err), 400, {"Content-Type": "application/json"}

        # Load historical data and extract feature schema
        hist_df, feature_cols = load_hist_and_features()
        news_items = load_recent_news(limit=30)

        # Generate scenario (forward rows)
        scenario_rows = call_llm_for_scenario(
            feature_cols=feature_cols,
            hist_sample_df=hist_df,
            news_items=news_items,
            scenario_text=scenario_text,
            horizon_weeks=horizon_weeks,
        )

        scenario_df = pd.DataFrame(scenario_rows)
        if "date" not in scenario_df.columns:
            raise RuntimeError("Scenario rows are missing 'date' column.")

        # Retrieve current deployed model metadata
        deployment_meta = get_current_deployment()
        algorithm = deployment_meta["algorithm"]
        artifact_path = deployment_meta["artifact_path"]

        model = None
        baseline_value = None

        # Load model based on algorithm type
        if algorithm == "base":
            from joblib import load as joblib_load

            storage_client = storage.Client()
            path = artifact_path.replace("gs://", "")
            parts = path.split("/", 1)
            bucket_name, blob_path = parts
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            buf = io.BytesIO()
            blob.download_to_file(buf)
            buf.seek(0)

            baseline_model = joblib_load(buf)
            baseline_value = baseline_model.get("value", 0.0)

        else:
            model = load_model_from_gcs(artifact_path)

        # Ensure all model feature columns exist in scenario_df
        for c in feature_cols:
            if c not in scenario_df.columns:
                scenario_df[c] = np.nan

        # Ensure 'rn' exists as an index
        if "rn" in feature_cols:
            scenario_df["rn"] = np.arange(len(scenario_df))

        X = scenario_df[feature_cols].copy()
        X = X.fillna(0.0)

        # Predict values
        if algorithm == "base":
            if baseline_value is None:
                baseline_value = 0.0
            preds = np.full(len(X), baseline_value)
        else:
            preds = model.predict(X)

        # Build prediction output
        result_df = pd.DataFrame(
            {
                "date": scenario_df["date"],
                "predicted_delinquency_rate": preds,
            }
        )
        result_df["week"] = scenario_df.get("week", None)
        result_df["year"] = scenario_df.get("year", None)

        response = {
            "deployment_id": deployment_meta["deployment_id"],
            "model_version_id": deployment_meta["model_version_id"],
            "algorithm": algorithm,
            "scenario_rows": scenario_rows,
            "predictions": result_df.to_dict(orient="records"),
            "horizon_weeks": horizon_weeks,
            "feature_cols": feature_cols,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        return (
            json.dumps(response, ensure_ascii=False, default=str),
            200,
            {"Content-Type": "application/json"},
        )

    except Exception as e:
        err = {"error": "Internal error", "details": str(e)}
        print(f"ml-generate-scenario error: {err}")
        return json.dumps(err), 500, {"Content-Type": "application/json"}