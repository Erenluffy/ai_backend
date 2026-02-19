#!/bin/bash
set -e

echo "🚀 Starting Ollama and Flask backend..."

# Verify Ollama installation
echo "🔍 Verifying Ollama installation..."
if [ -f /usr/local/bin/ollama ]; then
    echo "✅ Ollama binary found at /usr/local/bin/ollama"
    ls -la /usr/local/bin/ollama
else
    echo "❌ Ollama binary not found"
fi

# Start Ollama in background
if [ -f /usr/local/bin/ollama ]; then
    echo "📦 Starting Ollama server..."
    OLLAMA_HOST=0.0.0.0:11434 /usr/local/bin/ollama serve &
    OLLAMA_PID=$!
    
    echo "⏳ Waiting for Ollama to initialize..."
    sleep 10
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is running!"
        
        # Pull smaller model (change this to phi, tinyllama, or gemma:2b)
        echo "📥 Pulling $OLLAMA_MODEL model (size: ~1.5GB)..."
        /usr/local/bin/ollama pull $OLLAMA_MODEL
        echo "✅ Model pulled successfully!"
        
        # Verify model is available
        echo "📋 Available models:"
        /usr/local/bin/ollama list
    else
        echo "⚠️  Ollama not responding"
    fi
else
    echo "⚠️  Ollama not installed - running Flask only"
fi

# Start Flask app
# Start Flask app with gunicorn - use PORT from environment
echo "🌐 Starting Flask backend on port ${PORT:-5000}..."
echo "Using API Provider: $API_PROVIDER"
echo "Using Model: $OLLAMA_MODEL"
echo "Ollama URL: $OLLAMA_URL"

# Run gunicorn with dynamic port
exec gunicorn \
    --bind 0.0.0.0:${PORT:-5000} \
    --workers 1 \
    --threads 2 \
    --timeout 300 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    app:app
