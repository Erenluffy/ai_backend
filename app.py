from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
from datetime import datetime
import os
import logging
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS for production
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost", "http://127.0.0.1", "http://your-domain.com", "*"],
        "methods": ["GET", "POST", "OPTIONS", "DELETE"],
        "allow_headers": ["Content-Type"]
    }
})

# Store conversations (use Redis in production)
conversations = {}

# API Configuration
API_PROVIDER = os.getenv('API_PROVIDER', 'ollama').lower()  # deepseek, openai, anthropic, gemini, ollama
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# Ollama configuration
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')  # or mistral, codellama, phi, neural-chat, etc.

# DeepSeek configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Gemini configuration
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

def get_ai_response(message, conversation_history=[]):
    """Get response from configured AI API"""
    
    if API_PROVIDER == 'deepseek':
        return get_deepseek_response(message, conversation_history)
    elif API_PROVIDER == 'openai':
        return get_openai_response(message, conversation_history)
    elif API_PROVIDER == 'anthropic':
        return get_anthropic_response(message, conversation_history)
    elif API_PROVIDER == 'gemini':
        return get_gemini_response(message, conversation_history)
    elif API_PROVIDER == 'ollama':
        return get_ollama_response(message, conversation_history)
    else:
        return f"Error: Unknown API provider '{API_PROVIDER}'. Please set API_PROVIDER to 'deepseek', 'openai', 'anthropic', 'gemini', or 'ollama'."

def get_ollama_response(message, conversation_history=[]):
    """Get response from local Ollama (completely free)"""
    
    # Build context from conversation history
    messages = []
    
    # Add system message
    messages.append({"role": "system", "content": "You are a helpful AI assistant."})
    
    # Add conversation history
    for entry in conversation_history[-5:]:  # Last 5 exchanges for context
        if entry.get('user'):
            messages.append({"role": "user", "content": entry['user']})
        if entry.get('bot'):
            messages.append({"role": "assistant", "content": entry['bot']})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Ollama API payload
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 500
        }
    }
    
    try:
        logger.info(f"Sending request to Ollama with model: {OLLAMA_MODEL}")
        
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=60  # Ollama might take longer for first response
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'message' in data and 'content' in data['message']:
                return data['message']['content'].strip()
            else:
                logger.error(f"Unexpected Ollama response format: {data}")
                return "Error: Unexpected response format from Ollama"
        else:
            logger.error(f"Ollama API error {response.status_code}: {response.text}")
            
            if response.status_code == 404:
                return f"Error: Model '{OLLAMA_MODEL}' not found. Please pull it first with: ollama pull {OLLAMA_MODEL}"
            elif response.status_code == 500:
                return "Error: Ollama server error. Make sure Ollama is running properly."
            else:
                return f"Error {response.status_code}: Could not get response from Ollama"
            
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. Is it running?")
        return "Error: Cannot connect to Ollama. Please make sure Ollama is running (run 'ollama serve' in terminal)"
    except requests.exceptions.Timeout:
        logger.error("Ollama request timeout")
        return "Error: Ollama request timed out. The model might still be loading."
    except Exception as e:
        logger.error(f"Unexpected error with Ollama: {str(e)}")
        return f"Error with Ollama: {str(e)}"

def get_deepseek_response(message, conversation_history=[]):
    """Get response from DeepSeek API"""
    if not DEEPSEEK_API_KEY:
        return "Error: DeepSeek API key not configured. Please set DEEPSEEK_API_KEY environment variable."
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Prepare messages with conversation history
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    
    # Add conversation history if needed
    for entry in conversation_history[-5:]:  # Last 5 exchanges for context
        messages.append({"role": "user", "content": entry.get('user', '')})
        messages.append({"role": "assistant", "content": entry.get('bot', '')})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.7,
        "stream": False
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Extract the response text
        if 'choices' in data and len(data['choices']) > 0:
            return data['choices'][0]['message']['content']
        else:
            logger.error(f"Unexpected DeepSeek response format: {data}")
            return "Error: Unexpected response format from DeepSeek API"
            
    except requests.exceptions.RequestException as e:
        logger.error(f"DeepSeek API error: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Response: {e.response.text}")
        return f"Error connecting to DeepSeek API: {str(e)}"

def get_openai_response(message, conversation_history=[]):
    """Get response from OpenAI API"""
    if not OPENAI_API_KEY:
        return "Error: OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
    
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        
        # Prepare messages with conversation history
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        
        # Add conversation history if needed
        for entry in conversation_history[-5:]:  # Last 5 exchanges for context
            messages.append({"role": "user", "content": entry.get('user', '')})
            messages.append({"role": "assistant", "content": entry.get('bot', '')})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return f"Error getting OpenAI response: {str(e)}"

def get_anthropic_response(message, conversation_history=[]):
    """Get response from Anthropic Claude API"""
    if not ANTHROPIC_API_KEY:
        return "Error: Anthropic API key not configured. Please set ANTHROPIC_API_KEY environment variable."
    
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Prepare messages with conversation history
    messages = []
    
    # Add system message
    system_message = "You are a helpful AI assistant."
    
    # Add conversation history
    for entry in conversation_history[-5:]:
        if entry.get('user'):
            messages.append({"role": "user", "content": entry['user']})
        if entry.get('bot'):
            messages.append({"role": "assistant", "content": entry['bot']})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 500,
        "temperature": 0.7,
        "system": system_message,
        "messages": messages
    }
    
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        if 'content' in data and len(data['content']) > 0:
            # Extract text from content blocks
            content = data['content'][0]
            if content['type'] == 'text':
                return content['text']
            else:
                return str(content)
        else:
            return "Error: Unexpected response format from Anthropic API"
            
    except Exception as e:
        logger.error(f"Anthropic API error: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Response: {e.response.text}")
        return f"Error getting Anthropic response: {str(e)}"

def get_gemini_response(message, conversation_history=[]):
    """Get response from Google Gemini API"""
    if not GEMINI_API_KEY:
        return "Error: Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
    
    # Build context from conversation history
    context = ""
    for entry in conversation_history[-3:]:  # Last 3 exchanges for context
        if entry.get('user'):
            context += f"User: {entry['user']}\n"
        if entry.get('bot'):
            context += f"Assistant: {entry['bot']}\n"
    
    # Create prompt with history and current message
    prompt = f"{context}User: {message}\nAssistant:"
    
    # Try the most reliable endpoint first
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    
    # Correct payload format for Gemini API
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 500,
            "topP": 0.95,
            "topK": 40
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        logger.info(f"Sending request to Gemini API with prompt: {prompt[:50]}...")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        # Log response status for debugging
        logger.info(f"Gemini API response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Gemini API response received")
            
            # Extract text from response
            try:
                if 'candidates' in data and len(data['candidates']) > 0:
                    candidate = data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            return parts[0]['text'].strip()
                
                # If we can't find the text, return the whole response for debugging
                logger.error(f"Unexpected response format: {data}")
                return f"Error: Unexpected response format. Please check logs."
                
            except Exception as e:
                logger.error(f"Error parsing Gemini response: {str(e)}")
                return f"Error parsing response: {str(e)}"
        
        elif response.status_code == 403:
            error_msg = "API key is invalid or doesn't have access to Gemini API. Please check:"
            error_msg += "\n1. Your API key is correct"
            error_msg += "\n2. Gemini API is enabled in Google Cloud Console"
            error_msg += "\n3. Billing is enabled (if required)"
            logger.error(f"403 Forbidden: {response.text}")
            return error_msg
            
        elif response.status_code == 429:
            return "Error: Rate limit exceeded or quota exhausted. Please try again later."
            
        else:
            # Try to parse error message
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg = error_data['error'].get('message', 'Unknown error')
                    logger.error(f"Gemini API error: {error_msg}")
                    return f"Gemini API error: {error_msg}"
            except:
                pass
            
            logger.error(f"Gemini API error {response.status_code}: {response.text}")
            return f"Error {response.status_code}: Could not get response from Gemini API"
            
    except requests.exceptions.Timeout:
        logger.error("Gemini API timeout")
        return "Error: Request to Gemini API timed out"
    except requests.exceptions.ConnectionError:
        logger.error("Connection error to Gemini API")
        return "Error: Could not connect to Gemini API"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "AI Chatbot API is running",
        "version": "1.0.0",
        "api_provider": API_PROVIDER,
        "container": True
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "api_provider": API_PROVIDER
    }), 200

@app.route('/api/config')
def get_config():
    """Get current API configuration"""
    return jsonify({
        "api_provider": API_PROVIDER,
        "deepseek_configured": bool(DEEPSEEK_API_KEY),
        "openai_configured": bool(OPENAI_API_KEY),
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "ollama_configured": True,  # Ollama doesn't need an API key
        "ollama_model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_URL
    })

@app.route('/api/ollama/models', methods=['GET'])
def list_ollama_models():
    """List available Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return jsonify({"success": True, "models": models})
        else:
            return jsonify({"success": False, "error": "Could not fetch models"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
            
        message = data.get('message', '').strip()
        if not message:
            return jsonify({"success": False, "error": "Message required"}), 400
        
        conversation_id = data.get('conversation_id', str(uuid.uuid4()))
        
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        # Get conversation history for context
        history = conversations[conversation_id]
        
        # Get AI response from configured provider
        ai_message = get_ai_response(message, history)
        
        # Store conversation
        conversations[conversation_id].append({
            "user": message,
            "bot": ai_message,
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify({
            "success": True,
            "message": ai_message,
            "conversation_id": conversation_id,
            "api_provider": API_PROVIDER
        })
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/history/<conversation_id>', methods=['GET'])
def get_history(conversation_id):
    if conversation_id in conversations:
        return jsonify({
            "success": True,
            "history": conversations[conversation_id]
        })
    return jsonify({
        "success": False,
        "error": "Conversation not found"
    }), 404

@app.route('/api/clear/<conversation_id>', methods=['DELETE'])
def clear_conversation(conversation_id):
    if conversation_id in conversations:
        conversations[conversation_id] = []
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Conversation not found"}), 404

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', '0') == '1'
    
    logger.info(f"Starting Flask app on port {port}, debug={debug}")
    logger.info(f"Using API provider: {API_PROVIDER}")
    
    # Validate API key based on provider
    if API_PROVIDER == 'deepseek' and not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY not set! DeepSeek API will not work.")
    elif API_PROVIDER == 'openai' and not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set! OpenAI API will not work.")
    elif API_PROVIDER == 'anthropic' and not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set! Anthropic API will not work.")
    elif API_PROVIDER == 'gemini' and not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set! Gemini API will not work.")
    elif API_PROVIDER == 'ollama':
        logger.info(f"Ollama configured with model: {OLLAMA_MODEL}, URL: {OLLAMA_URL}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
