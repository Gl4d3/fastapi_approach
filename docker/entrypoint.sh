#!/bin/bash

# Exit on any error
set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Wait for Redis if configured
if [ ! -z "$REDIS_HOST" ]; then
    log "Waiting for Redis at $REDIS_HOST:$REDIS_PORT..."
    while ! nc -z $REDIS_HOST $REDIS_PORT; do
        sleep 1
    done
    log "Redis is ready!"
fi

# Create necessary directories
mkdir -p /app/uploads /app/logs /app/templates

# Set proper permissions
chmod 755 /app/uploads /app/logs /app/templates

# Check if Tesseract is available
if command -v tesseract >/dev/null 2>&1; then
    log "Tesseract OCR is available: $(tesseract --version | head -1)"
else
    log "ERROR: Tesseract OCR is not installed!"
    exit 1
fi

# Start the application
log "Starting Kenyan Document Processing API..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1