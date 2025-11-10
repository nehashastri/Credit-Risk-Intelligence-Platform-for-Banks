from google.cloud import bigquery
import functions_framework

# Initialize BigQuery client
client = bigquery.Client()

# SQL queries
fact_all_indicators_weekly = """
WITH fred_weekly AS (
  SELECT EXTRACT(WEEK FROM date) AS week, EXTRACT(YEAR FROM date) AS year, 
  AVG(marketyield2yr) AS marketyield2yr, AVG(marketyield10yr) AS marketyield10yr, 
  AVG(inflationrate) AS inflationrate, AVG(mortgagerate30yr) AS mortgagerate30yr
  FROM pipeline-882-team-project.landing.fact_macro_indicators_daily
  GROUP BY week, year
  ORDER BY year, week
), 
fred_monthly AS (
  SELECT EXTRACT(WEEK FROM date) AS week, EXTRACT(YEAR FROM date) AS year, 
  AVG(fedfundrate) AS fedfundrate, AVG(cpiurban) AS cpiurban, 
  AVG(unemployrate) AS unemployrate, AVG(ownedconsumercredit) AS ownedconsumercredit, 
  AVG(recessionindicator) AS recessionindicator, AVG(realgdp) AS realgdp
  FROM pipeline-882-team-project.landing.fact_macro_indicators_monthly
  GROUP BY week, year
  ORDER BY year, week
), 
yfinance_weekly AS (
  SELECT EXTRACT(WEEK FROM date) AS week, EXTRACT(YEAR FROM date) AS year, 
  AVG(consumerdisc_price) AS consumerdisc_price, AVG(consumerdisc_vol) AS consumerdisc_vol, 
  AVG(consumerstaple_price) AS consumerstaple_price, AVG(consumerstaple_vol) AS consumerstaple_vol,
  AVG(financial_price) AS financial_price, AVG(financial_vol) AS financial_vol, 
  AVG(tech_price) AS tech_price, AVG(tech_vol) AS tech_vol, 
  AVG(energy_price) AS energy_price, AVG(energy_vol) AS energy_vol, 
  AVG(industrial_price) AS industrial_price, AVG(industrial_vol) AS industrial_vol,
  AVG(utilities_price) AS utilities_price, AVG(utilities_vol) AS utilities_vol, 
  AVG(health_price) AS health_price, AVG(health_vol) AS health_vol, 
  AVG(material_price) AS material_price, AVG(material_vol) AS material_vol, 
  AVG(comm_price) AS comm_price, AVG(comm_vol) AS comm_vol
  FROM pipeline-882-team-project.landing.yfinance_table
  GROUP BY week, year
  ORDER BY year, week
),
credit_delinq_quarterly AS (
  SELECT EXTRACT(WEEK FROM date) AS week, EXTRACT(YEAR FROM date) AS year, 
  AVG(delinq) AS delinq
  FROM pipeline-882-team-project.landing.fact_credit_outcomes
  GROUP BY week, year
  ORDER BY year, week
),
combined AS (
  SELECT * FROM fred_weekly
  LEFT JOIN fred_monthly USING (week, year)
  LEFT JOIN yfinance_weekly USING (week, year)
  LEFT JOIN credit_delinq_quarterly USING (week, year)
),
with_boundaries AS (
  SELECT *,
    LAST_VALUE(fedfundrate IGNORE NULLS) OVER (ORDER BY year, week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS fedfundrate_filled,
    LAST_VALUE(unemployrate IGNORE NULLS) OVER (ORDER BY year, week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS unemployrate_filled,
    LAST_VALUE(recessionindicator IGNORE NULLS) OVER (ORDER BY year, week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS recessionindicator_filled,
    LAST_VALUE(cpiurban IGNORE NULLS) OVER (ORDER BY year, week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cpiurban_prev,
    FIRST_VALUE(cpiurban IGNORE NULLS) OVER (ORDER BY year, week ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS cpiurban_next,
    LAST_VALUE(ownedconsumercredit IGNORE NULLS) OVER (ORDER BY year, week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS ownedconsumercredit_prev,
    FIRST_VALUE(ownedconsumercredit IGNORE NULLS) OVER (ORDER BY year, week ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS ownedconsumercredit_next,
    LAST_VALUE(realgdp IGNORE NULLS) OVER (ORDER BY year, week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS realgdp_prev,
    FIRST_VALUE(realgdp IGNORE NULLS) OVER (ORDER BY year, week ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS realgdp_next,
    LAST_VALUE(delinq IGNORE NULLS) OVER (ORDER BY year, week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS delinq_prev,
    FIRST_VALUE(delinq IGNORE NULLS) OVER (ORDER BY year, week ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS delinq_next,
    SUM(CASE WHEN cpiurban IS NOT NULL THEN 1 ELSE 0 END) OVER (ORDER BY year, week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cpiurban_grp,
    SUM(CASE WHEN ownedconsumercredit IS NOT NULL THEN 1 ELSE 0 END) OVER (ORDER BY year, week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS ownedconsumercredit_grp,
    SUM(CASE WHEN realgdp IS NOT NULL THEN 1 ELSE 0 END) OVER (ORDER BY year, week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS realgdp_grp,
    SUM(CASE WHEN delinq IS NOT NULL THEN 1 ELSE 0 END) OVER (ORDER BY year, week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS delinq_grp
  FROM combined
),
with_group_position AS (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY cpiurban_grp ORDER BY year, week) - 1 AS cpiurban_pos,
    COUNT(*) OVER (PARTITION BY cpiurban_grp) AS cpiurban_total,
    ROW_NUMBER() OVER (PARTITION BY ownedconsumercredit_grp ORDER BY year, week) - 1 AS ownedconsumercredit_pos,
    COUNT(*) OVER (PARTITION BY ownedconsumercredit_grp) AS ownedconsumercredit_total,
    ROW_NUMBER() OVER (PARTITION BY realgdp_grp ORDER BY year, week) - 1 AS realgdp_pos,
    COUNT(*) OVER (PARTITION BY realgdp_grp) AS realgdp_total,
    ROW_NUMBER() OVER (PARTITION BY delinq_grp ORDER BY year, week) - 1 AS delinq_pos,
    COUNT(*) OVER (PARTITION BY delinq_grp) AS delinq_total
  FROM with_boundaries
), 
final AS (
  SELECT 
  week, year,
  COALESCE(delinq, delinq_prev + (delinq_next - delinq_prev) * delinq_pos / delinq_total) AS delinq,
  marketyield2yr, marketyield10yr, inflationrate, mortgagerate30yr,
  fedfundrate_filled AS fedfundrate,
  COALESCE(cpiurban, cpiurban_prev + (cpiurban_next - cpiurban_prev) * cpiurban_pos / cpiurban_total) AS cpiurban,
  unemployrate_filled AS unemployrate,
  COALESCE(ownedconsumercredit, ownedconsumercredit_prev + (ownedconsumercredit_next - ownedconsumercredit_prev) * ownedconsumercredit_pos / ownedconsumercredit_total) AS ownedconsumercredit,
  recessionindicator_filled AS recessionindicator,
  COALESCE(realgdp, realgdp_prev + (realgdp_next - realgdp_prev) * realgdp_pos / realgdp_total) AS realgdp,
  consumerdisc_price, consumerdisc_vol, consumerstaple_price, consumerstaple_vol,
  financial_price, financial_vol, tech_price, tech_vol, energy_price, energy_vol,
  industrial_price, industrial_vol, utilities_price, utilities_vol, health_price, health_vol,
  material_price, material_vol, comm_price, comm_vol
  FROM with_group_position
  ORDER BY year, week
)
SELECT * FROM final
WHERE week != 0 and week != 53
AND year >= 1999
"""

@functions_framework.http
def create_ml_dataset(request):
    """
    Cloud Function that creates the gold tables if missing, then loads data into them.
    """
    PROJECT_ID = "pipeline-882-team-project"
    DATASET_ID = "gold"

    tables = {
        "fact_all_indicators_weekly": fact_all_indicators_weekly,
    }

    full_dataset_id = f"{PROJECT_ID}.{DATASET_ID}"

    # Ensure dataset exists
    try:
        client.get_dataset(full_dataset_id)
        print(f"Dataset {full_dataset_id} already exists.")
    except Exception:
        dataset = bigquery.Dataset(full_dataset_id)
        dataset.location = "US"  # choose your location
        client.create_dataset(dataset)
        print(f"Created dataset {full_dataset_id}.")

    dataset_ref = client.dataset(DATASET_ID, project=PROJECT_ID)

    try:    
        for table_name, query in tables.items():
            table_ref = dataset_ref.table(table_name)

            # Check if table exists
            try:
                client.get_table(table_ref)
                print(f"Table '{table_name}' already exists.")
            except Exception:
                # Create an empty table if it doesn't exist
                schema = []  # BigQuery can infer schema from the query
                table = bigquery.Table(table_ref, schema=schema)
                client.create_table(table)
                print(f"Created table '{table_name}' in dataset '{DATASET_ID}'.")

            # Now run the query and overwrite
            job_config = bigquery.QueryJobConfig(
                destination=f"{PROJECT_ID}.{DATASET_ID}.{table_name}",
                write_disposition="WRITE_TRUNCATE"
            )

            query_job = client.query(query, job_config=job_config)
            query_job.result()
            print(f"Loaded data into {table_name} successfully.")

        return ("All gold tables created/loaded successfully.", 200)

    except Exception as e:
        print(f"Error loading tables: {e}")
        return (f"Error loading tables: {str(e)}", 500)