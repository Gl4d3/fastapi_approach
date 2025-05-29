#!/bin/bash

# Test script
set -e

echo "Running tests for Kenyan Document Processing System..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run tests
echo "Running pytest..."
pytest tests/ -v --cov=app --cov-report=html --cov-report=term

echo "Tests complete!"
echo "Coverage report available in htmlcov/index.html"