# Dockerfile for Credit Risk Intelligence Platform
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# --- FIX FOR SECRETS.TOML LOCATION ---
# Ensure the .streamlit directory exists and copy secrets.toml if it's nested under /streamlit/.streamlit
RUN mkdir -p /app/.streamlit && \
    if [ -f /app/streamlit/.streamlit/secrets.toml ]; then \
        cp /app/streamlit/.streamlit/secrets.toml /app/.streamlit/secrets.toml ; \
    fi && \
    chmod 600 /app/.streamlit/secrets.toml || true

# Environment variables (Cloud Run expects port 8080)
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose Streamlit port
EXPOSE 8080

# Health check (optional, Cloud Run compatible)
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "Home.py"]
