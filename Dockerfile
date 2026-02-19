# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies including wget
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama - Multiple methods for reliability
RUN set -ex && \
    # Method 1: Try official install script first
    (curl -fsSL https://ollama.com/install.sh | sh) || \
    # Method 2: Try direct download with wget
    (wget -q https://ollama.com/download/ollama-linux-amd64 -O /usr/local/bin/ollama && \
     chmod +x /usr/local/bin/ollama) || \
    # Method 3: Try alternative download with curl
    (curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64 -o /usr/local/bin/ollama && \
     chmod +x /usr/local/bin/ollama) || \
    # Method 4: Download specific version as last resort
    (curl -L https://github.com/ollama/ollama/releases/download/v0.1.46/ollama-linux-amd64 -o /usr/local/bin/ollama && \
     chmod +x /usr/local/bin/ollama)

# Verify the installation
RUN /usr/local/bin/ollama --version || echo "Ollama installation may have issues"

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

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
    echo "✅ Ollama binary found"\n\
    /usr/local/bin/ollama --version 2>/dev/null || echo "⚠️  Version check failed"\n\
else\n\
    echo "❌ Ollama binary not found"\n\
fi\n\
\n\
# Start Ollama in background if binary exists\n\
if [ -f /usr/local/bin/ollama ]; then\n\
    echo "📦 Starting Ollama server..."\n\
    /usr/local/bin/ollama serve &\n\
    OLLAMA_PID=$!\n\
    \n\
    # Wait for Ollama to start\n\
    echo "⏳ Waiting for Ollama to initialize..."\n\
    sleep 15\n\
    \n\
    # Check if Ollama is running\n\
    if curl -s http://localhost:11434/api/tags > /dev/null; then\n\
        echo "✅ Ollama is running!"\n\
        \n\
        # Pull the model if not exists\n\
        echo "📥 Checking for model: $OLLAMA_MODEL"\n\
        if ! /usr/local/bin/ollama list | grep -q "$OLLAMA_MODEL"; then\n\
            echo "⬇️  Pulling $OLLAMA_MODEL model (this may take 5-10 minutes)..."\n\
            /usr/local/bin/ollama pull $OLLAMA_MODEL\n\
            echo "✅ Model $OLLAMA_MODEL pulled successfully!"\n\
        else\n\
            echo "✅ Model $OLLAMA_MODEL already exists"\n\
        fi\n\
        \n\
        # List available models\n\
        echo "📋 Available models:"\n\
        /usr/local/bin/ollama list\n\
    else\n\
        echo "⚠️  Ollama not responding"\n\
    fi\n\
else\n\
    echo "⚠️  Ollama not installed - running Flask only"\n\
fi\n\
\n\
# Start Flask app with gunicorn\n\
echo "🌐 Starting Flask backend on port 5000..."\n\
echo "Using API Provider: $API_PROVIDER"\n\
echo "Using Model: $OLLAMA_MODEL"\n\
\n\
# Run gunicorn\n\
exec gunicorn \\\n\
    --bind 0.0.0.0:5000 \\\n\
    --workers 2 \\\n\
    --threads 4 \\\n\
    --timeout 300 \\\n\
    --log-level info \\\n\
    --access-logfile - \\\n\
    --error-logfile - \\\n\
    app:app\n\
' > /app/start.sh && chmod +x /app/start.sh

# Set Ollama environment variables
ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MODELS=/root/.ollama/models \
    OLLAMA_KEEP_ALIVE=0 \
    API_PROVIDER=ollama \
    OLLAMA_MODEL=mistral

# Create Ollama models directory
RUN mkdir -p /root/.ollama/models

# Expose ports
EXPOSE 5000 11434

# Set entrypoint
ENTRYPOINT ["/app/start.sh"]
