# functions/landing-load-news/main.py
import os
import functions_framework
from google.cloud import bigquery

# Load .env file during local development; use --set-env-vars in deployment
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Environment Variables ----------
GCP_PROJECT_ID  = os.getenv("GCP_PROJECT_ID")                  # e.g. pipeline-882-team-project
RAW_DATASET     = os.getenv("RAW_DATASET", "raw")
RAW_TABLE       = os.getenv("RAW_TABLE", "raw_news_articles")  # Must match the standard table name used by the raw function
LANDING_DATASET = os.getenv("LANDING_DATASET", "landing")

RAW_FQN      = f"`{GCP_PROJECT_ID}.{RAW_DATASET}.{RAW_TABLE}`"
LANDING_ART  = f"`{GCP_PROJECT_ID}.{LANDING_DATASET}.news_articles`"
LANDING_FACT = f"`{GCP_PROJECT_ID}.{LANDING_DATASET}.fact_news_relevance_scores`"

# ---------- DDL: Ensure Schema and Tables Exist ----------
DDL_SQL = f"""
CREATE SCHEMA IF NOT EXISTS `{GCP_PROJECT_ID}.{LANDING_DATASET}`;

CREATE TABLE IF NOT EXISTS {LANDING_ART} (
  article_id STRING,
  topic STRING,
  title STRING,
  url STRING,
  source_domain STRING,
  published_at TIMESTAMP,
  score FLOAT64,
  snippet STRING,
  ingest_datetime TIMESTAMP,
  ingest_date DATE
)
PARTITION BY DATE(published_at)
CLUSTER BY topic, source_domain;

CREATE TABLE IF NOT EXISTS {LANDING_FACT} (
  date DATE NOT NULL,
  relevance_fed_policy FLOAT64,
  relevance_cpi FLOAT64,
  relevance_labor FLOAT64,
  relevance_markets FLOAT64,
  relevance_energy FLOAT64,
  relevance_real_estate FLOAT64,
  relevance_layoff FLOAT64,
  num_articles_total INT64,
  ingest_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  notes STRING
)
PARTITION BY date;
"""

# ---------- MERGE: raw -> landing.news_articles ----------
# Normalize published_at (handle string/NULL values): if SAFE_CAST fails, fall back to ingest_datetime
MERGE_LANDING_SQL = f"""
MERGE {LANDING_ART} T
USING (
  SELECT
    GENERATE_UUID() AS article_id,
    topic,
    title,
    url,
    source_domain,
    COALESCE(SAFE_CAST(published_at AS TIMESTAMP), ingest_datetime) AS published_at,
    CAST(score AS FLOAT64) AS score,
    snippet,
    ingest_datetime,
    ingest_date
  FROM {RAW_FQN}
) S
ON T.url = S.url
   AND (T.published_at IS NOT DISTINCT FROM S.published_at)
WHEN MATCHED THEN UPDATE SET
  T.title           = COALESCE(S.title, T.title),
  T.topic           = COALESCE(S.topic, T.topic),
  T.source_domain   = COALESCE(S.source_domain, T.source_domain),
  T.score           = COALESCE(S.score, T.score),
  T.snippet         = COALESCE(S.snippet, T.snippet),
  T.ingest_datetime = S.ingest_datetime,
  T.ingest_date     = S.ingest_date,
  T.published_at    = S.published_at
WHEN NOT MATCHED THEN
  INSERT ROW;
"""

# ---------- Aggregation: Last 7 Days (Handles NULLs and Topic Variations Safely) ----------
AGG_WEEK_SQL = f"""
DECLARE _start DATE DEFAULT DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY);
DECLARE _end   DATE DEFAULT CURRENT_DATE();

CREATE TEMP TABLE _a AS
SELECT
  COALESCE(DATE(published_at), ingest_date) AS date_key,  -- Handle NULL values safely
  CASE
    WHEN LOWER(topic) IN ('layoff','layoffs')                  THEN 'layoff'
    WHEN LOWER(topic) IN ('fed_policy','fed','fomc')           THEN 'fed_policy'
    WHEN LOWER(topic) IN ('cpi','inflation')                   THEN 'cpi'
    WHEN LOWER(topic) IN ('labor','employment','jobs')         THEN 'labor'
    WHEN LOWER(topic) IN ('markets','market')                  THEN 'markets'
    WHEN LOWER(topic) IN ('energy','oil','gas')                THEN 'energy'
    WHEN LOWER(topic) IN ('real_estate','housing')             THEN 'real_estate'
    ELSE LOWER(topic)
  END AS topic_norm,
  AVG(CAST(score AS FLOAT64)) AS avg_rel,
  COUNT(*) AS cnt
FROM {LANDING_ART}
WHERE COALESCE(DATE(published_at), ingest_date) BETWEEN _start AND _end
GROUP BY 1,2;

IF (SELECT COUNT(*) FROM _a)=0 THEN
  SELECT 'no rows to aggregate for last 7 days (after normalization)' AS msg;
ELSE
  MERGE {LANDING_FACT} T
  USING (
    SELECT
      date_key AS date,
      MAX(IF(topic_norm='fed_policy',  avg_rel, NULL))  AS relevance_fed_policy,
      MAX(IF(topic_norm='cpi',         avg_rel, NULL))  AS relevance_cpi,
      MAX(IF(topic_norm='labor',       avg_rel, NULL))  AS relevance_labor,
      MAX(IF(topic_norm='markets',     avg_rel, NULL))  AS relevance_markets,
      MAX(IF(topic_norm='energy',      avg_rel, NULL))  AS relevance_energy,
      MAX(IF(topic_norm='real_estate', avg_rel, NULL))  AS relevance_real_estate,
      MAX(IF(topic_norm='layoff',      avg_rel, NULL))  AS relevance_layoff,
      SUM(cnt) AS num_articles_total,
      CURRENT_TIMESTAMP() AS ingest_timestamp
    FROM _a
    GROUP BY date_key
  ) S
  ON T.date = S.date
  WHEN MATCHED THEN UPDATE SET
    T.relevance_fed_policy  = S.relevance_fed_policy,
    T.relevance_cpi         = S.relevance_cpi,
    T.relevance_labor       = S.relevance_labor,
    T.relevance_markets     = S.relevance_markets,
    T.relevance_energy      = S.relevance_energy,
    T.relevance_real_estate = S.relevance_real_estate,
    T.relevance_layoff      = S.relevance_layoff,
    T.num_articles_total    = S.num_articles_total,
    T.ingest_timestamp      = S.ingest_timestamp
  WHEN NOT MATCHED THEN INSERT
    (date, relevance_fed_policy, relevance_cpi, relevance_labor,
     relevance_markets, relevance_energy, relevance_real_estate,
     relevance_layoff, num_articles_total, ingest_timestamp)
  VALUES
    (S.date, S.relevance_fed_policy, S.relevance_cpi, S.relevance_labor,
     S.relevance_markets, S.relevance_energy, S.relevance_real_estate,
     S.relevance_layoff, S.num_articles_total, S.ingest_timestamp);
END IF;
"""

def _run_query(client: bigquery.Client, sql: str):
    job = client.query(sql)
    job.result()

@functions_framework.http
def landing_load_news(request):
    if not GCP_PROJECT_ID:
        return ("Missing env: GCP_PROJECT_ID", 500)

    client = bigquery.Client(project=GCP_PROJECT_ID)
    try:
        _run_query(client, DDL_SQL)            # 1) Ensure DDL (schema and tables)
        _run_query(client, MERGE_LANDING_SQL)  # 2) Upsert from raw → landing (with published_at normalization)
        _run_query(client, AGG_WEEK_SQL)       # 3) Aggregate last 7 days
        return ("landing load done (last 7 days, normalized)", 200)
    except Exception as e:
        return (f"landing load failed: {e}", 500)