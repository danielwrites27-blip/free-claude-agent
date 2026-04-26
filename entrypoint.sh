#!/bin/bash
# Fix volume permissions then drop to appuser
chown -R appuser:appuser /app/data 2>/dev/null || true
export SENTENCE_TRANSFORMERS_HOME=/app/data/sentence_transformers

# Run model health check at startup (background) + schedule daily at 09:00 UTC
gosu appuser python /app/src/model_health_check.py >> /app/data/health_check.log 2>&1 &
(crontab -l 2>/dev/null; echo "0 9 * * * gosu appuser python /app/src/model_health_check.py >> /app/data/health_check.log 2>&1") | crontab -

exec gosu appuser python /app/app.py
