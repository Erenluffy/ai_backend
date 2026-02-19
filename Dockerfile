# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies (split into multiple steps for better debugging)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama using the official install script with error handling
RUN curl -fsSL https://ollama.com/install.sh | sh || \
    (echo "Failed to install Ollama via script, trying alternative method..." && \
     curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/local/bin/ollama && \
     chmod +x /usr/local/bin/ollama)

# Copy requirements first (for better caching)
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
# Start Ollama in background\n\
echo "📦 Starting Ollama server..."\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
# Wait for Ollama to start\n\
echo "⏳ Waiting for Ollama to initialize..."\n\
sleep 10\n\
\n\
# Check if Ollama is running\n\
if ! curl -s http://localhost:11434/api/tags > /dev/null; then\n\
    echo "❌ Ollama failed to start!"\n\
    exit 1\n\
fi\n\
\n\
echo "✅ Ollama is running!"\n\
\n\
# Pull the model if not exists\n\
echo "📥 Checking for model: $OLLAMA_MODEL"\n\
if ! ollama list | grep -q "$OLLAMA_MODEL"; then\n\
    echo "⬇️  Pulling $OLLAMA_MODEL model (this may take 5-10 minutes)..."\n\
    ollama pull $OLLAMA_MODEL\n\
    echo "✅ Model $OLLAMA_MODEL pulled successfully!"\n\
else\n\
    echo "✅ Model $OLLAMA_MODEL already exists"\n\
fi\n\
\n\
# List available models\n\
echo "📋 Available models:"\n\
ollama list\n\
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
    OLLAMA_KEEP_ALIVE=0

# Create Ollama models directory
RUN mkdir -p /root/.ollama/models

# Expose ports
EXPOSE 5000 11434

# Set entrypoint
ENTRYPOINT ["/app/start.sh"]
