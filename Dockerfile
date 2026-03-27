# ============================================================
# Stage 1: Builder – install Python dependencies
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies for PyMuPDF, torch, and other native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies into a dedicated prefix so we can copy
# the wheel cache to the final image cleanly.
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2: Final runtime image
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create required data directories
RUN mkdir -p \
    data/papers \
    data/chroma_db \
    data/lora_adapters \
    data/cache \
    logs

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Streamlit port
EXPOSE 8501
# Ollama API port (proxied / informational)
EXPOSE 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "ui/web.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
