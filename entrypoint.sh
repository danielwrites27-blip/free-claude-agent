#!/bin/bash
# Fix volume permissions then drop to appuser
chown -R appuser:appuser /app/data 2>/dev/null || true
exec gosu appuser python /app/app.py
