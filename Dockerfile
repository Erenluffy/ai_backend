# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies including file command
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    wget \
    file \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and install Ollama properly
RUN set -ex && \
    wget -q https://github.com/ollama/ollama/releases/download/v0.1.46/ollama-linux-amd64 -O /tmp/ollama && \
    file /tmp/ollama | grep -q "ELF" || (echo "Downloaded file is not a valid binary" && exit 1) && \
    install -o root -g root -m 0755 /tmp/ollama /usr/local/bin/ollama && \
    rm /tmp/ollama

# Verify the installation
RUN /usr/local/bin/ollama --version || echo "Ollama version check failed, but binary exists"

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir gunicorn && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🚀 Starting Ollama and Flask backend..."\n\
\n\
# Verify Ollama installation\n\
echo "🔍 Verifying Ollama installation..."\n\
if [ -f /usr/local/bin/ollama ]; then\n\
    echo "✅ Ollama binary found at /usr/local/bin/ollama"\n\
    ls -la /usr/local/bin/ollama\n\
else\n\
    echo "❌ Ollama binary not found"\n\
fi\n\
\n\
# Start Ollama in background\n\
if [ -f /usr/local/bin/ollama ]; then\n\
    echo "📦 Starting Ollama server..."\n\
    OLLAMA_HOST=0.0.0.0:11434 /usr/local/bin/ollama serve &\n\
    OLLAMA_PID=$!\n\
    \n\
    echo "⏳ Waiting for Ollama to initialize..."\n\
    MAX_RETRIES=30\n\
    RETRY_COUNT=0\n\
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do\n\
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then\n\
            echo "✅ Ollama is running!"\n\
            break\n\
        fi\n\
        RETRY_COUNT=$((RETRY_COUNT + 1))\n\
        echo "⏳ Waiting for Ollama... ($RETRY_COUNT/$MAX_RETRIES)"\n\
        sleep 2\n\
    done\n\
    \n\
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then\n\
        echo "⚠️  Ollama failed to respond within timeout"\n\
        if kill -0 $OLLAMA_PID 2>/dev/null; then\n\
            echo "✅ Ollama process is still running"\n\
        else\n\
            echo "❌ Ollama process died"\n\
        fi\n\
    else\n\
        echo "📥 Checking for model: $OLLAMA_MODEL"\n\
        if ! /usr/local/bin/ollama list | grep -q "$OLLAMA_MODEL"; then\n\
            echo "⬇️  Pulling $OLLAMA_MODEL model (this may take 5-10 minutes)..."\n\
            /usr/local/bin/ollama pull $OLLAMA_MODEL\n\
            echo "✅ Model $OLLAMA_MODEL pulled successfully!"\n\
        else\n\
            echo "✅ Model $OLLAMA_MODEL already exists"\n\
        fi\n\
        echo "📋 Available models:"\n\
        /usr/local/bin/ollama list\n\
    fi\n\
else\n\
    echo "⚠️  Ollama not installed - running Flask only"\n\
fi\n\
\n\
# Start Flask app with gunicorn - use PORT from environment\n\
echo "🌐 Starting Flask backend on port ${PORT:-5000}..."\n\
echo "Using API Provider: $API_PROVIDER"\n\
echo "Using Model: $OLLAMA_MODEL"\n\
echo "Ollama URL: $OLLAMA_URL"\n\
\n\
# Run gunicorn with dynamic port\n\
exec gunicorn \\\n\
    --bind 0.0.0.0:${PORT:-5000} \\\n\
    --workers 2 \\\n\
    --threads 4 \\\n\
    --timeout 300 \\\n\
    --log-level info \\\n\
    --access-logfile - \\\n\
    --error-logfile - \\\n\
    app:app\n\
' > /app/start.sh && chmod +x /app/start.sh

# Set environment variables
ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MODELS=/root/.ollama/models \
    OLLAMA_KEEP_ALIVE=0 \
    API_PROVIDER=ollama \
    OLLAMA_MODEL=phi \
    OLLAMA_URL=http://localhost:11434

# Create Ollama models directory
RUN mkdir -p /root/.ollama/models

# Expose ports
EXPOSE 11434

# Set entrypoint
ENTRYPOINT ["/app/start.sh"]
