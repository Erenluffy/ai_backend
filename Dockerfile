# Use Python 3.9 as base
FROM python:3.9-slim

# Install Ollama
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && curl -fsSL https://ollama.ai/install.sh | sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    API_PROVIDER=ollama \
    OLLAMA_URL=http://localhost:11434 \
    OLLAMA_MODEL=mistral \
    OLLAMA_HOST=0.0.0.0:11434

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create startup script
RUN echo '#!/bin/bash\n\
# Start Ollama in background\n\
ollama serve &\n\
\n\
# Wait for Ollama to start\n\
sleep 5\n\
\n\
# Pull the model (if not exists)\n\
ollama pull $OLLAMA_MODEL\n\
\n\
# Start Flask app with gunicorn\n\
gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 4 --timeout 120 --log-level info app:app\n\
' > /app/start.sh && chmod +x /app/start.sh

# Create a directory for Ollama models (persistent volume recommended)
RUN mkdir -p /root/.ollama

# Expose ports
EXPOSE 5000 11434

# Start everything
CMD ["/app/start.sh"]
