FROM python:3.11-slim

# ── System Dependencies ────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libhdf5-dev \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ──────────────────────────────────────
WORKDIR /app

# ── Python Dependencies ────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── App Files ──────────────────────────────────────────────
COPY . .

# ── Create Required Directories ────────────────────────────
RUN mkdir -p logs models/saved database

# ── Environment ────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Expose Ports ───────────────────────────────────────────
EXPOSE 8000   
# FastAPI
EXPOSE 8501   
# Streamlit

# ── Health Check ───────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# ── Default: API Server ────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
