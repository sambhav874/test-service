#!/bin/bash

# Activate the virtual environment
source /opt/venv/bin/activate

# Start the Uvicorn server with environment variables
exec uvicorn main:app \
    --host ${API_HOST:-0.0.0.0} \
    --port ${API_PORT:-8000} \
    --workers ${API_WORKERS:-1} \
    