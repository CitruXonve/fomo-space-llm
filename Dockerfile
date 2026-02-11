# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Add the project root to the Python module search path
    PYTHONPATH=/app \
    POETRY_VERSION=2.0.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN poetry install --no-root --no-directory

# Copy application code
COPY . .

# Expose the FastAPI default port
EXPOSE 8000

# Run the prepare-server step and then start the FastAPI server
CMD ["sh", "-c", "make prepare-server && fastapi run src/main.py --host 0.0.0.0 --port 8000"]
