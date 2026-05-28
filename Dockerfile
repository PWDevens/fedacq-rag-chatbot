FROM python:3.10-slim

WORKDIR /code

# Install system deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy project files first (for editable install)
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.lock

# Expose API port
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

# Run the app
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app.wsgi:app"]

