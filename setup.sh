#!/bin/bash

echo "🚀 Setting up AI Chatbot with Ollama and Mistral 7B..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "📦 Ollama not found. Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "✅ Ollama is already installed"
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "🔄 Starting Ollama server..."
    ollama serve &
    sleep 5
else
    echo "✅ Ollama server is running"
fi

# Pull Mistral model
echo "📥 Pulling Mistral 7B model (this may take a few minutes)..."
ollama pull mistral

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Build and run the Docker container
echo "🐳 Building and starting Docker container..."
docker-compose up --build -d

echo "✅ Setup complete!"
echo "🌐 Backend running at: http://localhost:5000"
echo "📝 Check logs with: docker-compose logs -f"
echo "🛑 Stop with: docker-compose down"

# Test the connection
sleep 3
if curl -s http://localhost:5000/health &> /dev/null; then
    echo "✅ Backend is healthy and running!"
else
    echo "⚠️  Backend might still be starting. Check logs with: docker-compose logs -f"
fi
