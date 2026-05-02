#!/bin/bash
export SENTENCE_TRANSFORMERS_HOME=/app/data/sentence_transformers

# Run model health check at startup (background)
python /app/src/model_health_check.py &

exec python /app/app.py
