#!/bin/bash
set -e

# Config
REGION="us-east-1"
REPO_NAME="interviewbot-app"
IMAGE_TAG="latest"
PROFILE="hrbot" 

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --profile $PROFILE)

# Full ECR image name
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

echo "Logging in to ECR..."
aws ecr get-login-password --region $REGION --profile $PROFILE \
| docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "Building Docker image..."
docker build -t ${REPO_NAME}:${IMAGE_TAG} .

echo "Tagging image as $ECR_URI..."
docker tag ${REPO_NAME}:${IMAGE_TAG} $ECR_URI

echo "Pushing image to ECR..."
docker push $ECR_URI

echo "Pushed to: $ECR_URI"
