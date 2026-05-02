#!/bin/bash
# Fix persistent storage permissions (HF Spaces mounts /data as root)
mkdir -p /data/chromadb /data/sentence_transformers
chown -R $(id -u):$(id -g) /data 2>/dev/null || true

export SENTENCE_TRANSFORMERS_HOME=/data/sentence_transformers

# Run model health check at startup (background)
python /app/src/model_health_check.py &

exec python /app/app.py
