FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install serving dependencies, including all supported model families.
# Development plotting and test tooling stays outside the runtime image.
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements-runtime.txt

# Copy source code and project assets
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000 8501

# Default launch command runs FastAPI server
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
