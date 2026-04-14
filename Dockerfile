FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc gosu && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app.py .
COPY .env.example .env
# Don't copy test files into the image — evals run locally, not in the pod
COPY src/ ./src/
COPY app.py requirements.txt entrypoint.sh ./
# No COPY tests/ — intentionally excluded

RUN useradd -m appuser && mkdir -p /app/data/chromadb && chown -R appuser:appuser /app
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
USER root
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860')" || exit 1
CMD ["/entrypoint.sh"]
