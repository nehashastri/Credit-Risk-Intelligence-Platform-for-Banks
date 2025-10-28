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
  --set-secrets FRED_API_KEY=FRED_API_Key:latest \
  --set-env-vars BUCKET_NAME=group11-ba882-fall25-data \
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

echo "Deploying landing-load-fred..."
gcloud functions deploy landing-load-fred \
  --gen2 \
  --runtime python312 \
  --trigger-http \
  --entry-point task \
  --source ./landing-load-fred \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512MB

echo "Deploying raw-fetch-yfinance..."
gcloud functions deploy raw-fetch-yfinance \
    --gen2 \
    --runtime python312 \
    --trigger-http \
    --entry-point task \
    --source ./raw-fetch-yfinance \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 512MB

echo "Deploying raw-fetch-news..."
source ../.env
gcloud functions deploy raw-fetch-news \
  --gen2 \
  --runtime python312 \
  --trigger-http \
  --entry-point raw_fetch_news \
  --source ./raw-fetch-news \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GCP_PROJECT_ID=$GCP_PROJECT_ID,RAW_DATASET=$RAW_DATASET,RAW_TABLE=$RAW_TABLE,LANDING_DATASET=$LANDING_DATASET,TAVILY_API_KEY=$TAVILY_API_KEY \
  --memory 512MB
