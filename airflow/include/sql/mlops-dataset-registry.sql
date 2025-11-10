INSERT INTO `pipeline-882-team-project.mlops.dataset` (
  dataset_id,
  model_id,
  data_version,
  gcs_path,
  row_count,
  feature_count,
  created_at
)
VALUES (
  '{{ dataset_id }}',
  '{{ model_id }}',
  '{{ data_version }}',
  'pipeline-882-team-project.gold.fact_all_indicators_weekly',
  {{ row_count }},
  {{ feature_count }},
  CURRENT_TIMESTAMP()
);
