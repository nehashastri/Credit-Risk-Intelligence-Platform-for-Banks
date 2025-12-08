# functions/ml-generate-scenario/main.py

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

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "pipeline-882-team-project")
MLOPS_DATASET = "mlops"
GOLD_DATASET = "gold"

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-mini")

# Default bucket storing model artifacts (joblib files, etc.)
DEFAULT_BUCKET = os.getenv("MODEL_BUCKET", "pipeline-882-team-project-models")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def safe_to_float(x):
    """Safely convert a value to float, returning None on failure."""
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def get_bq():
    """Create a BigQuery client for the configured project."""
    return bigquery.Client(project=PROJECT_ID)


def get_openai_client():
    """Create an OpenAI client using the API key from environment."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=OPENAI_API_KEY)


def load_model_from_gcs(artifact_path: str):
    """
    Load a model from a GCS artifact path.
    Expected format: gs://bucket/path/to/model.pkl
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
    Fetch the most recent active deployment for 'credit_delinquency_model'.
    Returns deployment metadata including artifact_path and algorithm name.
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


# -----------------------------------------------------------------------------
# Historical data + features
# -----------------------------------------------------------------------------
def load_hist_and_features():
    """
    Read the last 24 rows from gold.fact_all_indicators_weekly
    and extract feature columns (excluding date/year/week/target columns).
    We first grab the latest 24 weeks (ORDER BY year DESC, week DESC),
    then sort them back to ascending order for the prompt.
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

    # sort back to ascending time order
    hist_df = hist_df.sort_values(["year", "week"]).reset_index(drop=True)

    # 🔹 ensure rn exists for compatibility with the trained model
    if "rn" not in hist_df.columns:
        hist_df["rn"] = np.arange(len(hist_df))

    # columns to exclude from feature set
    exclude_cols = {
        "date",
        "year",
        "week",
        "delinq",
        "target",
        "credit_delinquency_rate",
    }
    feature_cols = [c for c in hist_df.columns if c not in exclude_cols]

    # 🔹 make sure rn is included as a feature (model expects it)
    if "rn" not in feature_cols:
        feature_cols.append("rn")

    return hist_df, feature_cols


def load_recent_news(limit=30):
    """
    Load recent economic news from landing.news_articles (last 30 days by default).
    This is optional context for the LLM.
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


# -----------------------------------------------------------------------------
# LLM call to generate a scenario
# -----------------------------------------------------------------------------
def strip_markdown_code_fence(text: str) -> str:
    """
    Remove Markdown ```json ... ``` fences if the model wrapped the JSON.
    """
    t = text.strip()
    if t.startswith("```"):
        # remove opening ``` or ```json
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        # remove trailing ```
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
    Use OpenAI Chat Completions API to generate future weekly feature values.
    Returns a list of rows:
    [
      {"date": "YYYY-MM-DD", "<feature1>": 0.0, "<feature2>": 0.0, ...},
      ...
    ]
    """
    client = get_openai_client()

    # Take last few rows as context
    hist_sample = hist_sample_df.tail(8).to_dict(orient="records")

    news_text = "\n\n".join(
        f"[{n['published_at']}] {n['title']} (topic={n['topic']}, score={n['score']})\n{n['snippet']}"
        for n in news_items
    )

    system_prompt = """
You are a macroeconomic scenario generator for a credit risk model.
Your job is to generate plausible weekly values for the next few weeks.

Return ONLY a single valid JSON object, with NO explanation, NO markdown, NO backticks.
The format MUST be exactly:

{
  "rows": [
    {
      "date": "YYYY-MM-DD",
      "<feature1>": 0.0,
      "<feature2>": 0.0,
      ...
    },
    ...
  ]
}

Rules:
- Generate exactly N weekly rows (N will be specified in the user prompt).
- "date" must be a valid ISO date string (YYYY-MM-DD).
- Include ALL feature columns given in the schema.
- Use ONLY numeric values for features.
- Do NOT include any target or delinquency columns.
"""

    user_prompt = f"""
Weekly schema:
- date
- Features: {feature_cols}

Recent historical data (last few weeks):
{json.dumps(hist_sample, indent=2)}

Recent news (last 30 days):
{news_text}

User scenario:
\"\"\"{scenario_text}\"\"\"

Requirements:
- Generate {horizon_weeks} future weekly rows.
- Start from the next plausible week after the last historical date.
- Use realistic values based on historical trends and the scenario.
- Fill ALL feature columns listed above.

Return ONLY the JSON object with the shape:
{{
  "rows": [
    {{
      "date": "YYYY-MM-DD",
      "<feature1>": 0.0,
      "<feature2>": 0.0,
      ...
    }},
    ...
  ]
}}
No comments, no explanations, no markdown, no extra text.
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


# -----------------------------------------------------------------------------
# Cloud Function HTTP entry point
# -----------------------------------------------------------------------------
@functions_framework.http
def task(request):
    """
    HTTP Cloud Function: generate an LLM-based scenario and run model predictions.

    Expected JSON body:
    {
      "scenario_text": "If unemployment jumps to 6% ...",
      "horizon_weeks": 8
    }

    Response JSON:
    {
      "deployment_id": "...",
      "model_version_id": "...",
      "algorithm": "...",
      "scenario_rows": [...],
      "predictions": [
        {
          "date": "YYYY-MM-DD",
          "predicted_delinquency_rate": ...,
          "week": ...,
          "year": ...
        },
        ...
      ],
      "horizon_weeks": 8,
      "feature_cols": [...],
      "timestamp": "..."
    }
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

        # 1) Load historical data and feature schema
        hist_df, feature_cols = load_hist_and_features()
        news_items = load_recent_news(limit=30)

        # 2) Generate scenario rows via LLM
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

        # 3) Load current deployed model metadata
        deployment_meta = get_current_deployment()
        algorithm = deployment_meta["algorithm"]
        artifact_path = deployment_meta["artifact_path"]

        # 4) Load model (baseline or sklearn)
        model = None
        baseline_value = None

        if algorithm == "base":
            # Baseline constant model
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
            # Sklearn-type model
            model = load_model_from_gcs(artifact_path)

        # 5) Ensure all feature columns exist in scenario_df
        for c in feature_cols:
            if c not in scenario_df.columns:
                scenario_df[c] = np.nan

        # 🔹 ensure rn column exists and has a stable numeric index
        if "rn" in feature_cols:
            scenario_df["rn"] = np.arange(len(scenario_df))

        X = scenario_df[feature_cols].copy()
        X = X.fillna(0.0)

        # 6) Run predictions
        if algorithm == "base":
            if baseline_value is None:
                baseline_value = 0.0
            preds = np.full(len(X), baseline_value)
        else:
            preds = model.predict(X)

        # 7) Build prediction dataframe
        result_df = pd.DataFrame(
            {
                "date": scenario_df["date"],
                "predicted_delinquency_rate": preds,
            }
        )
        # Optional week/year if scenario_df already has them
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
        print(f"❌ ml-generate-scenario error: {err}")
        return json.dumps(err), 500, {"Content-Type": "application/json"}