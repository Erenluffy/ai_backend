#!/bin/bash

echo "🚀 Starting AI Chatbot with Ollama locally..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Please install Ollama first:"
    echo "curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "🔄 Starting Ollama server..."
    ollama serve &
    sleep 5
fi

# Check if Mistral model is available
if ! ollama list | grep -q "mistral"; then
    echo "📥 Pulling Mistral model (this may take a few minutes)..."
    ollama pull mistral
fi

# Set environment variables
export API_PROVIDER=ollama
export OLLAMA_MODEL=mistral
export FLASK_DEBUG=1

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Run the Flask app
echo "🌐 Starting Flask backend on http://localhost:5000"
python app.py
