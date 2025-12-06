# ---------------------------------------------------------
# Optimized Dockerfile for Credit Risk Intelligence Platform
# Works perfectly on Cloud Run / Docker without build errors
# ---------------------------------------------------------

FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install minimal system dependencies required by pandas/pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (for Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt \
    && rm -rf /root/.cache/pip

# Copy all application files
COPY . .

# --- FIX FOR SECRETS.TOML LOCATION ---
RUN mkdir -p /app/.streamlit && \
    if [ -f /app/streamlit/.streamlit/secrets.toml ]; then \
        cp /app/streamlit/.streamlit/secrets.toml /app/.streamlit/secrets.toml ; \
    fi && \
    chmod 600 /app/.streamlit/secrets.toml || true

# Environment variables (Streamlit Cloud Run configuration)
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create non-root user (Cloud Run best practice)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check (Cloud Run compatible)
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "Home.py"]
