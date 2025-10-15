#!/bin/bash
gcloud config set project pipeline-882-team-project

echo "Deploying raw-fetch-fred..."
gcloud functions deploy raw-fetch-fred \
  --gen2 \
  --runtime python312 \
  --trigger-http \
  --entry-point task \
  --source ./raw-fetch-fred \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars FRED_API_KEY=$FRED_API_KEY \
  --memory 512MB

echo "Deploying raw-upload-fred..."
gcloud functions deploy raw-upload-fred \
  --gen2 \
  --runtime python312 \
  --trigger-http \
  --entry-point task \
  --source ./raw-upload-fred \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512MB
