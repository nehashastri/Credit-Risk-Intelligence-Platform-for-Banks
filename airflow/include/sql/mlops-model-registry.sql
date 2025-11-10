INSERT INTO `pipeline-882-team-project.mlops.model` (
  model_id,
  model_name,
  created_at
)
VALUES (
  '{{ model_id }}',
  '{{ model_name }}',
  CURRENT_TIMESTAMP()
);
