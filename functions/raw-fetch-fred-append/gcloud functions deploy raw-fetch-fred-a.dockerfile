gcloud functions deploy raw-fetch-fred-append \
  --runtime=python311 \
  --trigger-http \
  --allow-unauthenticated \
  --region=us-central1 \
  --entry-point=task \
  --source=functions/raw-fetch-fred-append \
  --set-env-vars PROJECT_ID=pipeline-882-team-project,BUCKET_NAME=group11-ba882-fall25-data \
  --set-secrets FRED_API_KEY=FRED_API_Key:latest
