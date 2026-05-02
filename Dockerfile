FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app.py requirements.txt entrypoint.sh ./

RUN useradd -m appuser && mkdir -p /app/data/chromadb && mkdir -p /data/chromadb /data/sentence_transformers && chown -R appuser:appuser /app /data

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER appuser
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV SENTENCE_TRANSFORMERS_HOME=/app/data/sentence_transformers
EXPOSE 7860

CMD ["/entrypoint.sh"]
