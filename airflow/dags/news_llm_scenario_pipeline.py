# dags/news_llm_scenario_pipeline.py

from datetime import timedelta
import os
import json
import math

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

# Variable import (sdk / models)
try:
    from airflow.sdk import Variable as AirflowVariable
except ImportError:
    from airflow.models import Variable as AirflowVariable

from google.cloud import bigquery
from openai import OpenAI  # pip install openai


# -----------------------------
# Global settings
# -----------------------------
ET = pendulum.timezone("America/New_York")
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "pipeline-882-team-project")
BQ_CLIENT = bigquery.Client(project=PROJECT_ID)


# -----------------------------
# OpenAI client helper
# -----------------------------
def get_openai_client():
    """
    Create an OpenAI client using the API key stored in Airflow Variables.
    Compatible with both airflow.sdk.Variable and airflow.models.Variable.
    """
    try:
        api_key = AirflowVariable.get("OPENAI_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        raise RuntimeError("Airflow Variable 'OPENAI_API_KEY' is not set.")
    return OpenAI(api_key=api_key)


# -----------------------------
# Numeric helpers
# -----------------------------
def safe_to_float(x):
    """Convert value to float if possible; otherwise return None."""
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def population_std(values):
    """
    Compute population standard deviation for a list of floats.
    Returns 0.0 if len(values) <= 1.
    """
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(var)


# -----------------------------
# Main task
# -----------------------------
def generate_llm_scenarios(**context):
    """
    End-to-end LLM scenario pipeline:

    1) Read recent news from landing.news_articles (last 30 days)
    2) Read the last 24 weeks from gold.fact_all_indicators_weekly
    3) Call an LLM multiple times to generate 8-week forward macro scenarios
    4) Aggregate across runs to compute mean & std-dev scenarios
    5) Insert mean and std scenarios into dedicated BigQuery tables
    """

    # ------------------------------------
    # 1. Load recent 30 days of news
    # ------------------------------------
    # news_articles schema:
    # article_id, topic, title, url, source_domain,
    # published_at, score, snippet, ingest_datetime, ingest_date
    sql_news = """
    SELECT
      published_at,
      title,
      topic,
      score,
      snippet
    FROM `pipeline-882-team-project.landing.news_articles`
    WHERE published_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
    ORDER BY published_at DESC
    LIMIT 200
    """

    news_rows = list(BQ_CLIENT.query(sql_news).result())
    news_items = [
        {
            "published_at": str(r["published_at"]),
            "title": r["title"],
            "topic": r.get("topic") or "",
            "score": r.get("score"),
            # snippet을 요약/본문처럼 사용
            "summary": r.get("snippet") or "",
            "content": r.get("snippet") or "",
        }
        for r in news_rows
    ]

    # ------------------------------------
    # 2. Load last 24 weeks from GOLD table
    # ------------------------------------
    sql_hist = """
    WITH ordered AS (
      SELECT
        *,
        ROW_NUMBER() OVER (ORDER BY year, week) AS rn
      FROM `pipeline-882-team-project.gold.fact_all_indicators_weekly`
    ),
    counts AS (
      SELECT COUNT(*) AS total_rows FROM ordered
    )
    SELECT
      o.*
    FROM ordered o, counts c
    WHERE o.rn > c.total_rows - 24
    ORDER BY year, week
    """

    hist_rows = list(BQ_CLIENT.query(sql_hist).result())
    if not hist_rows:
        raise RuntimeError("No historical data found for the last 24 weeks.")

    # Use the first row to infer the feature schema
    sample = hist_rows[0]

    # Exclude non-feature columns; we want only input features
    feature_cols = [
        f
        for f in sample.keys()
        if f
        not in (
            "date",                      # scenario 테이블에서 partition key로 사용
            "target",
            "credit_delinquency_rate",
            "year",                      # 인덱스/메타데이터
            "week",                      # 인덱스/메타데이터
        )
    ]

    # ------------------------------------
    # 3. Prepare input text for the LLM
    # ------------------------------------
    # 최근 기사 50개까지 사용
    news_text = "\n\n".join(
        (
            f"[{n['published_at']}] {n['title']}\n"
            f"topic: {n['topic']} | score: {n['score']}\n"
            f"{n['summary']}"
        )
        for n in news_items[:50]
    )

    # Create OpenAI client at runtime
    client = get_openai_client()

    # LLM sampling configuration
    N_RUNS = 15          # number of repeated calls for stability analysis
    N_WEEKS = 8          # number of future weeks to generate

    def call_llm_once():
        """
        Call the LLM once with the scenario-generation prompt.
        Returns: list of row dicts, each row containing at least:
          - "date"
          - feature columns from feature_cols
        """
        system_prompt = """
You are a financial macro scenario generator.
You MUST always respond with ONLY a single valid JSON object and nothing else.
The JSON must have the form:
{
  "rows": [
    {"date": "...", "feature1": 0.0, "feature2": 0.0, ...},
    ...
  ]
}
"""

        user_prompt = f"""
You are given:
1) The schema of a weekly indicator table:
   - date (weekly)
   - features: {feature_cols}
2) The last 24 weeks of historical data (not shown in full).
3) A list of recent news articles (titles + topic + snippet + scores).

Task:
- Read the news and infer how macro and financial indicators might shift.
- Generate a scenario for the NEXT {N_WEEKS} WEEKS.
- For each week, output one row containing:
  - "date": Monday or a consistent weekly start in ISO format (YYYY-MM-DD)
  - ALL feature columns with realistic numeric values, consistent with the last 24 weeks.
- Do NOT output any target or delinquency rate.

Return ONLY JSON, no explanation text.
"""

        full_input = user_prompt + "\n\nRECENT NEWS:\n" + news_text

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_input},
            ],
            temperature=0.3,
        )

        content_text = resp.choices[0].message.content

        data = json.loads(content_text)
        rows = data.get("rows", [])
        if not rows:
            raise RuntimeError("LLM returned no scenario rows.")
        return rows

    # ------------------------------------
    # 4. Multi-run sampling for stability
    # ------------------------------------
    all_runs = []  # list of list[rows]

    for i in range(N_RUNS):
        try:
            rows = call_llm_once()
            all_runs.append(rows)
            print(f"[LLM] Run {i+1}/{N_RUNS} succeeded with {len(rows)} rows.")
        except Exception as e:
            print(f"[LLM] Run {i+1}/{N_RUNS} failed: {e}")

    if not all_runs:
        raise RuntimeError("All LLM runs failed; no scenario can be generated.")

    # Assume runs all return the same number of weeks (or close to it).
    horizon = min(len(r) for r in all_runs)
    if horizon <= 0:
        raise RuntimeError("LLM runs produced empty scenario lists.")

    # ------------------------------------
    # 5. Aggregate across runs (mean & std)
    # ------------------------------------
    mean_rows = []
    std_rows = []

    for week_idx in range(horizon):
        # Get a representative row to copy the date
        example_row = None
        for run_rows in all_runs:
            if len(run_rows) > week_idx:
                example_row = run_rows[week_idx]
                break

        if example_row is None:
            continue

        date_val = example_row.get("date")

        mean_row = {"date": date_val}
        std_row = {"date": date_val}

        for feat in feature_cols:
            values = []
            for run_rows in all_runs:
                if len(run_rows) <= week_idx:
                    continue
                v = run_rows[week_idx].get(feat)
                v_float = safe_to_float(v)
                if v_float is not None:
                    values.append(v_float)

            if not values:
                mean_row[feat] = None
                std_row[feat] = None
            else:
                mean_row[feat] = sum(values) / len(values)
                std_row[feat] = population_std(values)

        mean_rows.append(mean_row)
        std_rows.append(std_row)

    print(f"[LLM] Aggregated {len(all_runs)} runs over horizon={horizon} weeks.")

    # ------------------------------------
    # 6. Insert into BigQuery (mean & std tables)
    # ------------------------------------
    mean_table_id = (
        "pipeline-882-team-project.gold.fact_all_indicators_weekly_llm_scenario_mean"
    )
    std_table_id = (
        "pipeline-882-team-project.gold.fact_all_indicators_weekly_llm_scenario_std"
    )

    # Fetch table schemas to align field names
    mean_table = BQ_CLIENT.get_table(mean_table_id)
    std_table = BQ_CLIENT.get_table(std_table_id)

    mean_fields = [f.name for f in mean_table.schema]
    std_fields = [f.name for f in std_table.schema]

    # Build rows aligned with BigQuery schemas
    mean_rows_bq = []
    for r in mean_rows:
        obj = {}
        for f in mean_fields:
            obj[f] = r.get(f)
        mean_rows_bq.append(obj)

    std_rows_bq = []
    for r in std_rows:
        obj = {}
        for f in std_fields:
            obj[f] = r.get(f)
        std_rows_bq.append(obj)

    # Insert into BigQuery
    errors_mean = BQ_CLIENT.insert_rows_json(mean_table_id, mean_rows_bq)
    if errors_mean:
        raise RuntimeError(f"Error inserting rows into {mean_table_id}: {errors_mean}")

    errors_std = BQ_CLIENT.insert_rows_json(std_table_id, std_rows_bq)
    if errors_std:
        raise RuntimeError(f"Error inserting rows into {std_table_id}: {errors_std}")

    print(
        f"[LLM] Inserted {len(mean_rows_bq)} mean scenario rows into {mean_table_id} "
        f"and {len(std_rows_bq)} std scenario rows into {std_table_id}."
    )


# -----------------------------
# DAG Definition
# -----------------------------
with DAG(
    dag_id="news_llm_scenario_pipeline",
    description=(
        "Generate future macro scenarios from recent news using OpenAI LLM, "
        "including stability analysis via multiple runs."
    ),
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,  # manual trigger for now; can be scheduled later
    catchup=False,
    is_paused_upon_creation=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["news", "llm", "scenario", "credit-risk", "stability"],
) as dag:

    generate_scenarios = PythonOperator(
        task_id="generate_llm_scenarios",
        python_callable=generate_llm_scenarios,
        execution_timeout=timedelta(minutes=20),
    )