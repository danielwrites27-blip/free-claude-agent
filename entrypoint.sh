#!/bin/bash
# Fix volume permissions then drop to appuser
chown -R appuser:appuser /app/data 2>/dev/null || true
exec su -s /bin/bash appuser -c "python /app/app.py"
