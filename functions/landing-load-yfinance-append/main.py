from google.cloud import bigquery
import functions_framework
from datetime import datetime

# Initialize BigQuery client
client = bigquery.Client()

# SQL query to pivot raw.sector_equity_features_2 → landing.fact_sector_prices_volumes_append
fact_sector_prices_volumes_append = """
-- Transform and pivot raw YFinance data into a wide daily sector table
SELECT
  DATE(date) AS date,

  -- Consumer Discretionary (XLY)
  ROUND(MAX(IF(ticker = 'XLY', close_price, NULL)), 2)             AS consumerdisc_price,
  ROUND(MAX(IF(ticker = 'XLY', CAST(volume AS FLOAT64), NULL)), 2) AS consumerdisc_vol,

  -- Consumer Staples (XLP)
  ROUND(MAX(IF(ticker = 'XLP', close_price, NULL)), 2)             AS consumerstaple_price,
  ROUND(MAX(IF(ticker = 'XLP', CAST(volume AS FLOAT64), NULL)), 2) AS consumerstaple_vol,

  -- Financials (XLF)
  ROUND(MAX(IF(ticker = 'XLF', close_price, NULL)), 2)             AS financial_price,
  ROUND(MAX(IF(ticker = 'XLF', CAST(volume AS FLOAT64), NULL)), 2) AS financial_vol,

  -- Technology (XLK)
  ROUND(MAX(IF(ticker = 'XLK', close_price, NULL)), 2)             AS tech_price,
  ROUND(MAX(IF(ticker = 'XLK', CAST(volume AS FLOAT64), NULL)), 2) AS tech_vol,

  -- Energy (XLE)
  ROUND(MAX(IF(ticker = 'XLE', close_price, NULL)), 2)             AS energy_price,
  ROUND(MAX(IF(ticker = 'XLE', CAST(volume AS FLOAT64), NULL)), 2) AS energy_vol,

  -- Industrials (XLI)
  ROUND(MAX(IF(ticker = 'XLI', close_price, NULL)), 2)             AS industrial_price,
  ROUND(MAX(IF(ticker = 'XLI', CAST(volume AS FLOAT64), NULL)), 2) AS industrial_vol,

  -- Utilities (XLU)
  ROUND(MAX(IF(ticker = 'XLU', close_price, NULL)), 2)             AS utilities_price,
  ROUND(MAX(IF(ticker = 'XLU', CAST(volume AS FLOAT64), NULL)), 2) AS utilities_vol,

  -- Health Care (XLV)
  ROUND(MAX(IF(ticker = 'XLV', close_price, NULL)), 2)             AS health_price,
  ROUND(MAX(IF(ticker = 'XLV', CAST(volume AS FLOAT64), NULL)), 2) AS health_vol,

  -- Materials (XLB)
  ROUND(MAX(IF(ticker = 'XLB', close_price, NULL)), 2)             AS material_price,
  ROUND(MAX(IF(ticker = 'XLB', CAST(volume AS FLOAT64), NULL)), 2) AS material_vol,

  -- Communication Services (XLC)
  ROUND(MAX(IF(ticker = 'XLC', close_price, NULL)), 2)             AS comm_price,
  ROUND(MAX(IF(ticker = 'XLC', CAST(volume AS FLOAT64), NULL)), 2) AS comm_vol,

  CURRENT_TIMESTAMP() AS load_timestamp

FROM `pipeline-882-team-project.raw.yfinance_table`
GROUP BY date
ORDER BY date
"""

@functions_framework.http
def landing_load_yfinance_append(request):
    """
    Cloud Function that appends new daily data from raw.sector_equity_features_2
    into landing.fact_sector_prices_volumes_append.
    """
    PROJECT_ID = "pipeline-882-team-project"
    DATASET_ID = "landing"
    TABLE_NAME = "yfinance_table"

    try:
        full_table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"

        # Use WRITE_APPEND instead of TRUNCATE
        job_config = bigquery.QueryJobConfig(
            destination=full_table_id,
            write_disposition="WRITE_APPEND"
        )

        print(f"🚀 Starting append from raw.sector_equity_features_2 → {full_table_id}")
        query_job = client.query(fact_sector_prices_volumes_append, job_config=job_config)
        query_job.result()

        print(f"✅ Successfully appended new rows into {full_table_id}")
        return (f"Data appended successfully into {full_table_id}.", 200)

    except Exception as e:
        print(f"❌ Error loading table: {e}")
        return (f"Error loading table: {str(e)}", 500)
