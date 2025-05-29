#!/bin/bash

# Setup script for development environment
set -e

echo "Setting up Kenyan Document Processing System..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p uploads logs templates tests/sample_docs

# Copy environment file
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please edit .env file with your configuration"
fi

# Make scripts executable
chmod +x scripts/*.sh
chmod +x docker/entrypoint.sh

echo "Setup complete!"
echo "To start the development server:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run: python -m app.main"
echo "3. Visit: http://localhost:8000"