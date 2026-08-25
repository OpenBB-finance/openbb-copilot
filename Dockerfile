# Multi-stage Dockerfile for OpenBB Ada
ARG PLATFORM=linux/amd64

# ============================================
# Stage 1: Builder
# ============================================
FROM --platform=$PLATFORM python:3.12-slim AS python-base

# Set environment variables for build stage
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt" \
    VENV_PATH="/opt/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Install build dependencies (poppler-utils for PDF processing)
RUN apt-get update -y && apt-get upgrade -y \
    && apt-get install --no-install-recommends -y poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools

# ============================================
# Stage 2: Builder
# ============================================
FROM python-base AS python-builder

RUN apt-get update -y && apt-get upgrade -y \
    && apt-get install --no-install-recommends -y \
    curl git build-essential unzip

# Set working directory for dependency files copying
WORKDIR $PYSETUP_PATH

# Upgrade setuptools, pip and install poetry
RUN pip install --upgrade setuptools poetry

# Copy dependency files
COPY poetry.lock pyproject.toml ./

# Install Python dependencies (to VENV_PATH)
RUN poetry config installer.max-workers 10 \
    && poetry install --without dev,testing \
    --no-interaction --no-ansi --no-cache --no-root

RUN poetry run pip install --upgrade pip

# ============================================
# Stage 3: Runtime
# ============================================
FROM python-base AS runtime
COPY --from=python-builder $VENV_PATH $VENV_PATH

# Set working directory for the application
WORKDIR $PYSETUP_PATH/app

# Copy application files
COPY openbb_ada ./openbb_ada
COPY startup.sh ./startup.sh

# Application configuration for restricted environments (i.e. Snowflake)
# Pre-download tiktoken encoding files and copy snowflake.env only if it exists
COPY . /tmp/build-context
RUN if [ -f /tmp/build-context/snowflake.env ]; then \
        python -c "import tiktoken; tiktoken.get_encoding('o200k_base'); tiktoken.get_encoding('cl100k_base')" && \
        mv /tmp/build-context/snowflake.env $PYSETUP_PATH/app/.env; \
    fi
RUN rm -rf /tmp/build-context

# Set default port for the application
ENV PORT=7778

# Expose the port the app runs on
EXPOSE ${PORT}

# Command to run the application
CMD ["bash", "startup.sh"]
