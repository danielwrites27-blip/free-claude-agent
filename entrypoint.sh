#!/bin/bash
# Fix volume permissions then drop to appuser
chown -R appuser:appuser /app/data 2>/dev/null || true
export SENTENCE_TRANSFORMERS_HOME=/app/data/sentence_transformers

# Run model health check at startup (background, as appuser)
gosu appuser python /app/src/model_health_check.py >> /app/data/health_check.log 2>&1 &

exec gosu appuser python /app/app.py
