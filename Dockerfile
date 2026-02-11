# Builder stage - Use Python 3.11 slim image as base
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Set environment variables
ENV POETRY_VERSION=2.0.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

# Install system dependencies and Poetry
RUN apt-get update && apt-get install -y curl build-essential && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN poetry install --no-root --no-directory

# Production/Runtime stage
FROM python:3.11-slim AS runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

# Add the project root to the Python module search path
# PYTHONPATH=/app 

# Install make for Makefile
RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

# Copy only the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY . .

# Expose the FastAPI default port
EXPOSE 8000

# Run the prepare-server step and then start the FastAPI server
CMD ["sh", "-c", "make prepare-server && echo 'Awaiting server initialization...' && fastapi run src/main.py --host 0.0.0.0 --port 8000"]
