#!/bin/bash

# Deployment script
set -e

echo "Deploying Kenyan Document Processing System..."

# Build Docker image
echo "Building Docker image..."
docker build -f docker/Dockerfile -t kenyan-doc-processor:latest .

# Start services
echo "Starting services..."
docker-compose -f docker/docker-compose.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Health check
echo "Performing health check..."
curl -f http://localhost:8000/health || exit 1

echo "Deployment complete!"
echo "API available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"