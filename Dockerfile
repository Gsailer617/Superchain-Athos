# Build stage
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.4.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./
COPY requirements*.txt ./

# Install dependencies with ML support
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir transformers ray[tune] optuna scikit-learn

# Copy Node.js files and install dependencies
COPY package*.json ./
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm ci --only=production

# Development stage
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    isort \
    mypy \
    pylint \
    bandit \
    safety

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p artifacts cache data/monitoring

# Set environment variables
ENV PYTHONPATH=/app
ENV NODE_ENV=development

# Production stage
FROM python:3.10-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    xvfb \
    nodejs \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

# Set Chrome environment variables
ENV DISPLAY=:99
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Set working directory
WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /app/requirements.txt .
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p artifacts cache data/monitoring

# Set environment variables
ENV PYTHONPATH=/app \
    NODE_ENV=production \
    PYTORCH_NO_CUDA=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Security hardening
RUN apt-get update && apt-get install -y \
    ca-certificates \
    openssl \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Create non-root user with limited permissions
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app \
    && mkdir -p /home/appuser/.cache \
    && chown -R appuser:appuser /home/appuser

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8050/health || exit 1

# Start Xvfb and application with proper signal handling
CMD ["sh", "-c", "trap 'kill $(jobs -p)' TERM; Xvfb :99 -screen 0 1024x768x16 & python telegram_bot.py"] 