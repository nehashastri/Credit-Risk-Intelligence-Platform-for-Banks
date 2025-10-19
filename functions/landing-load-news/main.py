# functions/landing-load-news/main.py
import os
import functions_framework
from google.cloud import bigquery

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GCP_PROJECT_ID  = os.getenv("GCP_PROJECT_ID")          # e.g. pipeline-882-team-project
RAW_DATASET     = os.getenv("RAW_DATASET", "raw")
RAW_TABLE       = os.getenv("RAW_TABLE", "raw_news_articles")
LANDING_DATASET = os.getenv("LANDING_DATASET", "landing")

RAW_FQN         = f"`{GCP_PROJECT_ID}.{RAW_DATASET}.{RAW_TABLE}`"
LANDING_ART     = f"`{GCP_PROJECT_ID}.{LANDING_DATASET}.news_articles`"
LANDING_FACT    = f"`{GCP_PROJECT_ID}.{LANDING_DATASET}.fact_news_relevance_scores`"

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

MERGE_LANDING_SQL = f"""
MERGE {LANDING_ART} T
USING (
  SELECT
    GENERATE_UUID() AS article_id,        -- raw에 고유키가 없으므로 landing에서 생성
    topic,
    title,
    url,
    source_domain,
    published_at,
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
  T.ingest_date     = S.ingest_date
WHEN NOT MATCHED THEN
  INSERT ROW;
"""

AGG_TODAY_SQL = f"""
DECLARE _today DATE DEFAULT CURRENT_DATE();

CREATE TEMP TABLE _a AS
SELECT
  DATE(published_at) AS date,
  topic,
  AVG(score) AS avg_rel,
  COUNT(*)   AS cnt
FROM {LANDING_ART}
WHERE DATE(published_at) = _today
GROUP BY 1,2;

IF (SELECT COUNT(*) FROM _a)=0 THEN
  SELECT 'no rows to aggregate for ' || CAST(_today AS STRING) AS msg;
ELSE
  MERGE {LANDING_FACT} T
  USING (
    SELECT
      date,
      MAX(IF(topic='fed_policy',  avg_rel, NULL))  AS relevance_fed_policy,
      MAX(IF(topic='cpi',         avg_rel, NULL))  AS relevance_cpi,
      MAX(IF(topic='labor',       avg_rel, NULL))  AS relevance_labor,
      MAX(IF(topic='markets',     avg_rel, NULL))  AS relevance_markets,
      MAX(IF(topic='energy',      avg_rel, NULL))  AS relevance_energy,
      MAX(IF(topic='real_estate', avg_rel, NULL))  AS relevance_real_estate,
      MAX(IF(topic='layoff',      avg_rel, NULL))  AS relevance_layoff,
      SUM(cnt)                                      AS num_articles_total,
      CURRENT_TIMESTAMP()                            AS ingest_timestamp
    FROM _a
    GROUP BY date
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
  WHEN NOT MATCHED THEN
    INSERT ROW;
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
        # 1) DDL 보장
        _run_query(client, DDL_SQL)
        # 2) raw -> landing 업서트
        _run_query(client, MERGE_LANDING_SQL)
        # 3) 오늘자 집계 업서트
        _run_query(client, AGG_TODAY_SQL)
        return ("landing load done", 200)
    except Exception as e:
        return (f"landing load failed: {e}", 500)

