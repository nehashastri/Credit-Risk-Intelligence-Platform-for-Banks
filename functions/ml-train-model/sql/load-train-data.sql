-- ==========================================
-- Training data: all but last 24 weeks
-- ==========================================

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
WHERE o.rn <= c.total_rows - 24
ORDER BY year, week;
