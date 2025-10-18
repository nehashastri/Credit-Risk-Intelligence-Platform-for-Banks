from google.cloud import bigquery
import functions_framework

# Initialize BigQuery client
client = bigquery.Client()

# SQL queries
fact_macro_indicators_monthly = """-- Merge 5 series, inner-join GDPC1, 1980+, then linearly interpolate gaps
WITH
fed AS (
  SELECT DATE(date) AS date, MAX(value) AS fedfundrate
  FROM `pipeline-882-team-project.raw.fedfunds`
  GROUP BY date
),
cpi AS (
  SELECT DATE(date) AS date, value AS cpiurban
  FROM `pipeline-882-team-project.raw.cpiaucsl`
),
un AS (
  SELECT DATE(date) AS date, value AS unemployrate
  FROM `pipeline-882-team-project.raw.unrate`
),
tot AS (
  SELECT DATE(date) AS date, value AS ownedconsumercredit
  FROM `pipeline-882-team-project.raw.totalsl`
),
rec AS (
  SELECT DATE(date) AS date, value AS recessionindicator
  FROM `pipeline-882-team-project.raw.usrec`
),
gdp AS (
  SELECT DATE(date) AS date, value AS realgdp
  FROM `pipeline-882-team-project.raw.gdpc1`
),

base AS (
  SELECT
    date,
    fedfundrate,
    cpiurban,
    unemployrate,
    ownedconsumercredit,
    recessionindicator
  FROM fed
  FULL OUTER JOIN cpi USING (date)
  FULL OUTER JOIN un  USING (date)
  FULL OUTER JOIN tot USING (date)
  FULL OUTER JOIN rec USING (date)
  WHERE date >= DATE '1980-01-01'
),

joined AS (
  SELECT
    b.date,
    b.fedfundrate,
    b.cpiurban,
    b.unemployrate,
    b.ownedconsumercredit,
    b.recessionindicator,
    g.realgdp
  FROM base b
  INNER JOIN gdp g USING (date)
),

long AS (
  SELECT date, series, value
  FROM joined
  UNPIVOT(value FOR series IN (
    fedfundrate,
    cpiurban,
    unemployrate,
    ownedconsumercredit,
    recessionindicator,
    realgdp
  ))
),

-- ✅ IGNORE NULLS goes inside FIRST_VALUE / LAST_VALUE
neighbors AS (
  SELECT
    date,
    series,
    value AS original_value,

    LAST_VALUE(value IGNORE NULLS) OVER (
      PARTITION BY series ORDER BY date
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS prev_val,
    LAST_VALUE(IF(value IS NOT NULL, date, NULL) IGNORE NULLS) OVER (
      PARTITION BY series ORDER BY date
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS prev_date,

    FIRST_VALUE(value IGNORE NULLS) OVER (
      PARTITION BY series ORDER BY date
      ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
    ) AS next_val,
    FIRST_VALUE(IF(value IS NOT NULL, date, NULL) IGNORE NULLS) OVER (
      PARTITION BY series ORDER BY date
      ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
    ) AS next_date
  FROM long
),

filled AS (
  SELECT
    date,
    series,
    CASE
      WHEN original_value IS NOT NULL THEN original_value
      WHEN prev_date IS NOT NULL AND next_date IS NOT NULL AND next_date <> prev_date THEN
        prev_val + (next_val - prev_val)
        * SAFE_DIVIDE(DATE_DIFF(date, prev_date, DAY), DATE_DIFF(next_date, prev_date, DAY))
      WHEN prev_val IS NOT NULL THEN prev_val
      WHEN next_val IS NOT NULL THEN next_val
      ELSE NULL
    END AS value
  FROM neighbors
),

wide AS (
  SELECT
    date,
  ROUND(MAX(IF(series = 'fedfundrate',         value, NULL)), 2) AS fedfundrate,
  ROUND(MAX(IF(series = 'cpiurban',            value, NULL)), 2) AS cpiurban,
  ROUND(MAX(IF(series = 'unemployrate',        value, NULL)), 2) AS unemployrate,
  ROUND(MAX(IF(series = 'ownedconsumercredit', value, NULL)), 2) AS ownedconsumercredit,
  ROUND(MAX(IF(series = 'recessionindicator',  value, NULL)), 2) AS recessionindicator,
  ROUND(MAX(IF(series = 'realgdp',             value, NULL)), 2) AS realgdp
  FROM filled
  GROUP BY date
)

SELECT *
FROM wide
ORDER BY date;
"""

fact_macro_indicators_daily = """-- 1) One row per date in each source (no GROUP BY)
WITH s2 AS (
  SELECT date, marketyield2yr
  FROM (
    SELECT DATE(date) AS date,
           value      AS marketyield2yr,
           ROW_NUMBER() OVER (PARTITION BY DATE(date) ORDER BY DATE(date)) AS rn
    FROM `pipeline-882-team-project.raw.dgs2`
  )
  WHERE rn = 1
),
s10 AS (
  SELECT date, marketyield10yr
  FROM (
    SELECT DATE(date) AS date,
           value      AS marketyield10yr,
           ROW_NUMBER() OVER (PARTITION BY DATE(date) ORDER BY DATE(date)) AS rn
    FROM `pipeline-882-team-project.raw.dgs10`
  )
  WHERE rn = 1
),
inf AS (
  SELECT date, inflationrate
  FROM (
    SELECT DATE(date) AS date,
           value      AS inflationrate,
           ROW_NUMBER() OVER (PARTITION BY DATE(date) ORDER BY DATE(date)) AS rn
    FROM `pipeline-882-team-project.raw.t5yie`
  )
  WHERE rn = 1
),
m30 AS (
  SELECT date, mortgagerate30yr
  FROM (
    SELECT DATE(date) AS date,
           value      AS mortgagerate30yr,
           ROW_NUMBER() OVER (PARTITION BY DATE(date) ORDER BY DATE(date)) AS rn
    FROM `pipeline-882-team-project.raw.mortgage30us`
  )
  WHERE rn = 1
),

-- 2) Build daily calendar from 1980-01-01 to the max date across all series
bounds AS (
  SELECT
    DATE '1980-01-01' AS start_date,
    GREATEST(
      (SELECT MAX(date) FROM s2),
      (SELECT MAX(date) FROM s10),
      (SELECT MAX(date) FROM inf),
      (SELECT MAX(date) FROM m30)
    ) AS end_date
),
calendar AS (
  SELECT day AS date
  FROM bounds, UNNEST(GENERATE_DATE_ARRAY(start_date, end_date)) AS day
),

-- 3) Merge daily series + LEFT JOIN weekly mortgage
wide AS (
  SELECT
    c.date,
    s2.marketyield2yr,
    s10.marketyield10yr,
    inf.inflationrate,
    m30.mortgagerate30yr
  FROM calendar c
  LEFT JOIN s2  USING (date)
  LEFT JOIN s10 USING (date)
  LEFT JOIN inf USING (date)
  LEFT JOIN m30 USING (date)
),

-- 4) Interpolate missing values (linear; carry edges)
nbrs AS (
  SELECT
    date,

    marketyield2yr,
    LAST_VALUE(marketyield2yr IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s2_prev_val,
    LAST_VALUE(IF(marketyield2yr IS NOT NULL, date, NULL) IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s2_prev_date,
    FIRST_VALUE(marketyield2yr IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS s2_next_val,
    FIRST_VALUE(IF(marketyield2yr IS NOT NULL, date, NULL) IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS s2_next_date,

    marketyield10yr,
    LAST_VALUE(marketyield10yr IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s10_prev_val,
    LAST_VALUE(IF(marketyield10yr IS NOT NULL, date, NULL) IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s10_prev_date,
    FIRST_VALUE(marketyield10yr IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS s10_next_val,
    FIRST_VALUE(IF(marketyield10yr IS NOT NULL, date, NULL) IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS s10_next_date,

    inflationrate,
    LAST_VALUE(inflationrate IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS inf_prev_val,
    LAST_VALUE(IF(inflationrate IS NOT NULL, date, NULL) IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS inf_prev_date,
    FIRST_VALUE(inflationrate IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS inf_next_val,
    FIRST_VALUE(IF(inflationrate IS NOT NULL, date, NULL) IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS inf_next_date,

    mortgagerate30yr,
    LAST_VALUE(mortgagerate30yr IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS m30_prev_val,
    LAST_VALUE(IF(mortgagerate30yr IS NOT NULL, date, NULL) IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS m30_prev_date,
    FIRST_VALUE(mortgagerate30yr IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS m30_next_val,
    FIRST_VALUE(IF(mortgagerate30yr IS NOT NULL, date, NULL) IGNORE NULLS) OVER (
      ORDER BY date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS m30_next_date
  FROM wide
),
filled AS (
  SELECT
    date,
    -- 2Y
    CASE
      WHEN marketyield2yr IS NOT NULL THEN marketyield2yr
      WHEN s2_prev_date IS NOT NULL AND s2_next_date IS NOT NULL AND s2_next_date <> s2_prev_date THEN
        s2_prev_val + (s2_next_val - s2_prev_val)
        * SAFE_DIVIDE(DATE_DIFF(date, s2_prev_date, DAY), DATE_DIFF(s2_next_date, s2_prev_date, DAY))
      WHEN s2_prev_val IS NOT NULL THEN s2_prev_val
      WHEN s2_next_val IS NOT NULL THEN s2_next_val
      ELSE NULL
    END AS marketyield2yr,
    -- 10Y
    CASE
      WHEN marketyield10yr IS NOT NULL THEN marketyield10yr
      WHEN s10_prev_date IS NOT NULL AND s10_next_date IS NOT NULL AND s10_next_date <> s10_prev_date THEN
        s10_prev_val + (s10_next_val - s10_prev_val)
        * SAFE_DIVIDE(DATE_DIFF(date, s10_prev_date, DAY), DATE_DIFF(s10_next_date, s10_prev_date, DAY))
      WHEN s10_prev_val IS NOT NULL THEN s10_prev_val
      WHEN s10_next_val IS NOT NULL THEN s10_next_val
      ELSE NULL
    END AS marketyield10yr,
    -- 5Y inflation expectations
    CASE
      WHEN inflationrate IS NOT NULL THEN inflationrate
      WHEN inf_prev_date IS NOT NULL AND inf_next_date IS NOT NULL AND inf_next_date <> inf_prev_date THEN
        inf_prev_val + (inf_next_val - inf_prev_val)
        * SAFE_DIVIDE(DATE_DIFF(date, inf_prev_date, DAY), DATE_DIFF(inf_next_date, inf_prev_date, DAY))
      WHEN inf_prev_val IS NOT NULL THEN inf_prev_val
      WHEN inf_next_val IS NOT NULL THEN inf_next_val
      ELSE NULL
    END AS inflationrate,
    -- 30Y mortgage (weekly source)
    CASE
      WHEN mortgagerate30yr IS NOT NULL THEN mortgagerate30yr
      WHEN m30_prev_date IS NOT NULL AND m30_next_date IS NOT NULL AND m30_next_date <> m30_prev_date THEN
        m30_prev_val + (m30_next_val - m30_prev_val)
        * SAFE_DIVIDE(DATE_DIFF(date, m30_prev_date, DAY), DATE_DIFF(m30_next_date, m30_prev_date, DAY))
      WHEN m30_prev_val IS NOT NULL THEN m30_prev_val
      WHEN m30_next_val IS NOT NULL THEN m30_next_val
      ELSE NULL
    END AS mortgagerate30yr
  FROM nbrs
)
SELECT date,
  ROUND(marketyield2yr, 2)   AS marketyield2yr,
  ROUND(marketyield10yr, 2)  AS marketyield10yr,
  ROUND(inflationrate, 2)    AS inflationrate,
  ROUND(mortgagerate30yr, 2) AS mortgagerate30yr
FROM filled
ORDER BY date;
"""

fact_credit_outcomes = """SELECT
  DATE(date) AS date,
  ROUND(AVG(value),2) AS delinq
FROM `pipeline-882-team-project.raw.drcclacbs`
GROUP BY date;
"""

@functions_framework.http
def landing_load_fred(request):
    """
    Cloud Function that populates three landing tables:
      1. fact_macro_indicators_monthly
      2. fact_macro_indicators_daily
      3. fact_credit_outcomes
    """
    PROJECT_ID = "pipeline-882-team-project"
    DATASET_ID = "landing"

    tables = {
        "fact_macro_indicators_monthly": fact_macro_indicators_monthly,
        "fact_macro_indicators_daily": fact_macro_indicators_daily,
        "fact_credit_outcomes": fact_credit_outcomes,
    }

    try:
        for table_name, query in tables.items():
            full_table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

            job_config = bigquery.QueryJobConfig(
                destination=full_table_id,
                write_disposition="WRITE_TRUNCATE"  # Replace old data
            )

            query_job = client.query(query, job_config=job_config)
            query_job.result()  # Wait for completion

            print(f"✅ Loaded {table_name} successfully.")

        return ("All landing tables loaded successfully.", 200)

    except Exception as e:
        print(f"❌ Error loading tables: {e}")
        return (f"Error loading tables: {str(e)}", 500)
