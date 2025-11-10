#!/bin/bash
set -e  # Stop if any command fails

PROJECT_ID="pipeline-882-team-project"
SERVICE_NAME="streamlit-poc"
REGION="us-central1"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "======================================================"
echo "Setting active project: ${PROJECT_ID}"
echo "======================================================"
gcloud config set project $PROJECT_ID

echo "======================================================"
echo "Building Docker image"
echo "======================================================"
docker build --no-cache -t $IMAGE .

echo "======================================================"
echo "Pushing image to Google Container Registry"
echo "======================================================"
docker push $IMAGE

echo "======================================================"
echo "Deploying to Cloud Run"
echo "======================================================"
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 1Gi \
    --port 8080 \
    --timeout 900 \
    --set-secrets="/app/.streamlit/secrets.toml=streamlit-secrets:latest"

echo "======================================================"
echo "Deployment complete!"
echo "======================================================"
gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)'
