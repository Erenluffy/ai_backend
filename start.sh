#!/bin/bash
set -e

echo "🚀 Starting Ollama and Flask backend..."

# Start Ollama in background
echo "📦 Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to start
echo "⏳ Waiting for Ollama to initialize..."
sleep 10

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama failed to start!"
    exit 1
fi

echo "✅ Ollama is running!"

# Pull the model if not exists
echo "📥 Checking for model: $OLLAMA_MODEL"
if ! ollama list | grep -q "$OLLAMA_MODEL"; then
    echo "⬇️  Pulling $OLLAMA_MODEL model (this may take 5-10 minutes)..."
    ollama pull $OLLAMA_MODEL
    echo "✅ Model $OLLAMA_MODEL pulled successfully!"
else
    echo "✅ Model $OLLAMA_MODEL already exists"
fi

# List available models
echo "📋 Available models:"
ollama list

# Start Flask app with gunicorn
echo "🌐 Starting Flask backend on port 5000..."
echo "Using API Provider: $API_PROVIDER"
echo "Using Model: $OLLAMA_MODEL"

# Run gunicorn
exec gunicorn \
    --bind 0.0.0.0:5000 \
    --workers 2 \
    --threads 4 \
    --timeout 300 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    app:app
