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
API_PROVIDER = os.getenv('API_PROVIDER', 'deepseek').lower()  # deepseek, openai, anthropic, etc.
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

# DeepSeek configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def get_ai_response(message, conversation_history=[]):
    """Get response from configured AI API"""
    
    if API_PROVIDER == 'deepseek':
        return get_deepseek_response(message, conversation_history)
    elif API_PROVIDER == 'openai':
        return get_openai_response(message, conversation_history)
    elif API_PROVIDER == 'anthropic':
        return get_anthropic_response(message, conversation_history)
    else:
        return f"Error: Unknown API provider '{API_PROVIDER}'. Please set API_PROVIDER to 'deepseek', 'openai', or 'anthropic'."

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
    
    # Prepare conversation
    prompt = "Human: You are a helpful AI assistant.\n\n"
    
    # Add conversation history
    for entry in conversation_history[-5:]:
        if entry.get('user'):
            prompt += f"Human: {entry['user']}\n\n"
        if entry.get('bot'):
            prompt += f"Assistant: {entry['bot']}\n\n"
    
    # Add current message
    prompt += f"Human: {message}\n\nAssistant:"
    
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 500,
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": message}
        ]
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
            return data['content'][0]['text']
        else:
            return "Error: Unexpected response format from Anthropic API"
            
    except Exception as e:
        logger.error(f"Anthropic API error: {str(e)}")
        return f"Error getting Anthropic response: {str(e)}"

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
        "anthropic_configured": bool(ANTHROPIC_API_KEY)
    })

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
    
    app.run(host='0.0.0.0', port=port, debug=debug)
