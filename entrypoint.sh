#!/bin/bash
# Fix volume permissions
chown -R appuser:appuser /app/data 2>/dev/null || true
# Drop to appuser and run app
exec su-exec appuser python /app/app.py
