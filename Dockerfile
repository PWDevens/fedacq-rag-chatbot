FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# System deps (curl for healthcheck, git for any repo ops)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps using lock file (cacheable layer)
COPY requirements.lock .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.lock

# Install project
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir .

EXPOSE 7860

HEALTHCHECK CMD curl -f http://localhost:7860/health || exit 1

CMD ["hypercorn", "--bind", "0.0.0.0:7860", "src.app.asgi:app"]
