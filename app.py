from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
from datetime import datetime
import os
import logging
import requests
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS for production
# Comprehensive CORS configuration
CORS(app, 
     origins="*",  # Allow all origins for now
     methods=["GET", "HEAD", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept"],
     supports_credentials=True,
     max_age=3600)

# Also add explicit OPTIONS handler for all routes
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response
# Store conversations (use Redis in production)
conversations = {}

# API Configuration
API_PROVIDER = os.getenv('API_PROVIDER', 'ollama').lower()
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# Ollama configuration
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'tinyllama:1.1b')  # Default to phi now

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
    
    # Retry logic for Ollama
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending request to Ollama with model: {OLLAMA_MODEL} (attempt {attempt + 1}/{max_retries})")
            
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=120  # Increased timeout for first response
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'message' in data and 'content' in data['message']:
                    return data['message']['content'].strip()
                else:
                    logger.error(f"Unexpected Ollama response format: {data}")
                    return "Error: Unexpected response format from Ollama"
            elif response.status_code == 404:
                return f"Error: Model '{OLLAMA_MODEL}' not found. Please pull it first."
            elif response.status_code == 500:
                if attempt < max_retries - 1:
                    logger.warning(f"Ollama server error, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                return "Error: Ollama server error. Please try again."
            else:
                logger.error(f"Ollama API error {response.status_code}: {response.text}")
                return f"Error {response.status_code}: Could not get response from Ollama"
                
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                logger.warning(f"Cannot connect to Ollama, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            logger.error("Cannot connect to Ollama. Is it running?")
            return "Error: Cannot connect to Ollama. Please make sure Ollama is running."
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                logger.warning(f"Ollama request timeout, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            logger.error("Ollama request timeout")
            return "Error: Ollama request timed out. The model might still be loading."
        except Exception as e:
            logger.error(f"Unexpected error with Ollama: {str(e)}")
            return f"Error with Ollama: {str(e)}"
    
    return "Error: Max retries exceeded"

def get_deepseek_response(message, conversation_history=[]):
    """Get response from DeepSeek API"""
    if not DEEPSEEK_API_KEY:
        return "Error: DeepSeek API key not configured. Please set DEEPSEEK_API_KEY environment variable."
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    
    for entry in conversation_history[-5:]:
        if entry.get('user'):
            messages.append({"role": "user", "content": entry.get('user', '')})
        if entry.get('bot'):
            messages.append({"role": "assistant", "content": entry.get('bot', '')})
    
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
        
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        
        for entry in conversation_history[-5:]:
            messages.append({"role": "user", "content": entry.get('user', '')})
            messages.append({"role": "assistant", "content": entry.get('bot', '')})
        
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
    
    messages = []
    system_message = "You are a helpful AI assistant."
    
    for entry in conversation_history[-5:]:
        if entry.get('user'):
            messages.append({"role": "user", "content": entry['user']})
        if entry.get('bot'):
            messages.append({"role": "assistant", "content": entry['bot']})
    
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
    
    context = ""
    for entry in conversation_history[-3:]:
        if entry.get('user'):
            context += f"User: {entry['user']}\n"
        if entry.get('bot'):
            context += f"Assistant: {entry['bot']}\n"
    
    prompt = f"{context}User: {message}\nAssistant:"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    
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
        logger.info(f"Sending request to Gemini API")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            try:
                if 'candidates' in data and len(data['candidates']) > 0:
                    candidate = data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            return parts[0]['text'].strip()
                
                logger.error(f"Unexpected response format: {data}")
                return f"Error: Unexpected response format."
                
            except Exception as e:
                logger.error(f"Error parsing Gemini response: {str(e)}")
                return f"Error parsing response: {str(e)}"
        
        elif response.status_code == 403:
            return "Error: API key is invalid or doesn't have access to Gemini API."
        elif response.status_code == 429:
            return "Error: Rate limit exceeded. Please try again later."
        else:
            return f"Error {response.status_code}: Could not get response from Gemini API"
            
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
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
# Add this after all your route definitions (around line 350)
@app.route('/debug/routes')
def list_routes():
    """List all registered routes (debug only)"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'path': str(rule)
        })
    return jsonify({
        'total_routes': len(routes),
        'routes': routes,
        'api_provider': API_PROVIDER,
        'model': OLLAMA_MODEL
    })
@app.route('/health')
def health():
    # Check Ollama health
    ollama_status = "unknown"
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            ollama_status = "healthy"
        else:
            ollama_status = "unhealthy"
    except:
        ollama_status = "unreachable"
    
    return jsonify({
        "status": "healthy",
        "api_provider": API_PROVIDER,
        "ollama_status": ollama_status,
        "model": OLLAMA_MODEL
    }), 200

@app.route('/api/config')
def get_config():
    """Get current API configuration"""
    # Check Ollama models
    ollama_models = []
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            ollama_models = [model['name'] for model in data.get('models', [])]
    except:
        pass
    
    return jsonify({
        "api_provider": API_PROVIDER,
        "deepseek_configured": bool(DEEPSEEK_API_KEY),
        "openai_configured": bool(OPENAI_API_KEY),
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "ollama_configured": True,
        "ollama_model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_URL,
        "ollama_available_models": ollama_models
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

@app.route('/api/ollama/pull/<model_name>', methods=['POST'])
def pull_ollama_model(model_name):
    """Pull a new Ollama model"""
    try:
        # This is a long-running operation
        response = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": model_name},
            timeout=1  # Immediate response, pull happens in background
        )
        return jsonify({"success": True, "message": f"Pulling model {model_name} in background"})
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
        
        history = conversations[conversation_id]
        
        # Get AI response
        start_time = time.time()
        ai_message = get_ai_response(message, history)
        response_time = time.time() - start_time
        
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
            "api_provider": API_PROVIDER,
            "response_time_seconds": round(response_time, 2)
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

# Add this configuration that runs for both gunicorn and direct execution
port = int(os.getenv('PORT', 5000))
debug = os.getenv('FLASK_DEBUG', '0') == '1'

logger.info(f"Starting Flask app on port {port}, debug={debug}")
logger.info(f"Using API provider: {API_PROVIDER}")

if API_PROVIDER == 'ollama':
    logger.info(f"Ollama configured with model: {OLLAMA_MODEL}, URL: {OLLAMA_URL}")

# This only runs when executing directly (python app.py), not with gunicorn
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=debug)
