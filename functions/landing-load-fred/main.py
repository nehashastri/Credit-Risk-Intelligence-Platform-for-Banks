import functions_framework
from google.cloud import bigquery
from google.cloud import secretmanager
import json

# Settings
PROJECT_ID = "pipeline-882-team-project"
DATASET_ID = "landing"
TABLE_ID = "fact_macro_indicators"

# SQL query (your provided SELECT)
FACT_MACRO_SQL = """
SELECT 
  DISTINCT cpiaucsl.date, fedfunds.value AS fed_funds_rate, 
  dgs2.value AS yield_2yr, dgs10.value AS yield_10yr, dgs10.value - dgs2.value AS yield_spread_2_10,
  cpiaucsl.value AS cpi, unrate.value AS unemployment_rate, totalsl.value AS consumer_credit, 
  mortgage30us.value AS mortgage_30yr, gdpc1.value AS gdp_real, usrec.value AS recession_flag, 
  t5yie.value AS breakeven_inflation_5yr, '' AS provenance_info, CURRENT_DATETIME('America/New_York') AS last_update
FROM `pipeline-882-team-project.raw.fedfunds` AS fedfunds
INNER JOIN `pipeline-882-team-project.raw.dgs2` AS dgs2
  ON fedfunds.date = dgs2.date
INNER JOIN `pipeline-882-team-project.raw.dgs10` AS dgs10
  ON fedfunds.date = dgs10.date
INNER JOIN `pipeline-882-team-project.raw.cpiaucsl` AS cpiaucsl
  ON fedfunds.date = cpiaucsl.date
INNER JOIN `pipeline-882-team-project.raw.unrate` AS unrate
  ON fedfunds.date = unrate.date
INNER JOIN `pipeline-882-team-project.raw.totalsl` AS totalsl
  ON fedfunds.date = totalsl.date
INNER JOIN `pipeline-882-team-project.raw.mortgage30us` AS mortgage30us
  ON fedfunds.date = mortgage30us.date
INNER JOIN `pipeline-882-team-project.raw.gdpc1` AS gdpc1
  ON fedfunds.date = gdpc1.date
INNER JOIN `pipeline-882-team-project.raw.usrec` AS usrec
  ON fedfunds.date = usrec.date
INNER JOIN `pipeline-882-team-project.raw.t5yie` AS t5yie
  ON fedfunds.date = t5yie.date
ORDER BY cpiaucsl.date ASC
"""

@functions_framework.http
def task(request):
    """Cloud Function to extract from raw schema and load into landing.fact_macro_indicators"""

    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    # Step 1: Run the query and fetch results
    print("Running transformation query...")
    job = client.query(FACT_MACRO_SQL)
    results = list(job.result())
    print(f"Fetched {len(results)} rows from raw schema.")

    # Step 2: Create destination table if missing
    try:
        client.get_table(table_ref)
        print(f"Table {table_ref} exists.")
    except Exception:
        print(f"Table {table_ref} not found. Creating it.")
        schema = [
            bigquery.SchemaField("date", "DATE"),
            bigquery.SchemaField("fed_funds_rate", "FLOAT64"),
            bigquery.SchemaField("yield_2yr", "FLOAT64"),
            bigquery.SchemaField("yield_10yr", "FLOAT64"),
            bigquery.SchemaField("yield_spread_2_10", "FLOAT64"),
            bigquery.SchemaField("cpi", "FLOAT64"),
            bigquery.SchemaField("unemployment_rate", "FLOAT64"),
            bigquery.SchemaField("consumer_credit", "FLOAT64"),
            bigquery.SchemaField("mortgage_30yr", "FLOAT64"),
            bigquery.SchemaField("gdp_real", "FLOAT64"),
            bigquery.SchemaField("recession_flag", "FLOAT64"),
            bigquery.SchemaField("breakeven_inflation_5yr", "FLOAT64"),
            bigquery.SchemaField("provenance_info", "STRING"),
            bigquery.SchemaField("last_update", "DATETIME"),
        ]
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)
        print(f"Created {table_ref}.")

    # Step 3: Load or truncate logic
    if len(results) == 0:
        print("Query returned 0 rows — clearing landing table.")
        client.query(f"DELETE FROM `{table_ref}` WHERE TRUE").result()
    else:
        print(f"Loading {len(results)} rows into landing table.")
        load_job = client.load_table_from_query(
            FACT_MACRO_SQL,
            table_ref,
            job_config=bigquery.QueryJobConfig(
                write_disposition="WRITE_TRUNCATE"
            ),
        )
        load_job.result()
        print("Landing table successfully updated.")

    return ("Landing table refresh completed.", 200)