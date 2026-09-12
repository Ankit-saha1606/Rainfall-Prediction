# --- Build stage: install dependencies into an isolated layer ---
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Runtime stage: slim final image, no build toolchain ---
FROM python:3.11-slim

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Bring in pre-built dependencies from the builder stage
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY data/ ./data/

RUN mkdir -p /app/models /app/logs && chown -R appuser:appuser /app

USER appuser

ENV MODEL_DIR=/app/models \
    LOG_DIR=/app/logs \
    PORT=5000 \
    FLASK_DEBUG=false

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:5000/health').status==200 else sys.exit(1)"

# Gunicorn as the production WSGI server (dev Flask server is single-threaded
# and explicitly not meant for production use).
CMD ["gunicorn", "--chdir", "src", "--bind", "0.0.0.0:5000", \
     "--workers", "2", "--threads", "4", "--timeout", "60", \
     "--access-logfile", "-", "--error-logfile", "-", \
     "app:app"]
