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
API_PROVIDER = os.getenv('API_PROVIDER', 'gemini').lower()  # deepseek, openai, anthropic, gemini
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

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
    else:
        return f"Error: Unknown API provider '{API_PROVIDER}'. Please set API_PROVIDER to 'deepseek', 'openai', 'anthropic', or 'gemini'."

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
    
    # Try different model names and endpoints
    models_to_try = [
        "gemini-1.5-pro",  # Latest model
        "gemini-1.0-pro",   # Stable model
        "gemini-pro"        # Original model name
    ]
    
    # Build context from conversation history
    context = ""
    for entry in conversation_history[-3:]:  # Last 3 exchanges for context
        if entry.get('user'):
            context += f"User: {entry['user']}\n"
        if entry.get('bot'):
            context += f"Assistant: {entry['bot']}\n"
    
    # Create prompt with history and current message
    prompt = f"{context}User: {message}\nAssistant:"
    
    last_error = None
    
    # Try each model until one works
    for model in models_to_try:
        try:
            # Different endpoint formats to try
            endpoints_to_try = [
                f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent",
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                f"https://generativelanguage.googleapis.com/v1/models/{model}:generateText",
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateText"
            ]
            
            for endpoint in endpoints_to_try:
                url = f"{endpoint}?key={GEMINI_API_KEY}"
                
                # Try different payload formats
                payload_formats = [
                    # Format 1: Standard chat format
                    {
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
                    },
                    # Format 2: Simplified format
                    {
                        "prompt": {
                            "text": prompt
                        },
                        "temperature": 0.7,
                        "max_output_tokens": 500
                    },
                    # Format 3: Messages format
                    {
                        "messages": [
                            {
                                "role": "user",
                                "parts": [{"text": prompt}]
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.7,
                            "maxOutputTokens": 500
                        }
                    }
                ]
                
                for payload in payload_formats:
                    try:
                        logger.info(f"Trying Gemini model: {model}, endpoint: {endpoint}")
                        
                        headers = {
                            "Content-Type": "application/json"
                        }
                        
                        response = requests.post(url, headers=headers, json=payload, timeout=30)
                        
                        if response.status_code == 200:
                            data = response.json()
                            logger.info(f"Success with model: {model}")
                            
                            # Try to extract text from different response formats
                            text = extract_gemini_text(data)
                            if text:
                                return text
                            
                        elif response.status_code == 404:
                            continue  # Try next endpoint/model
                        else:
                            # Log other errors but continue trying
                            logger.warning(f"Gemini API error with {model}: {response.status_code} - {response.text}")
                            
                    except requests.exceptions.RequestException as e:
                        last_error = str(e)
                        continue
                        
        except Exception as e:
            last_error = str(e)
            continue
    
    # If all attempts fail, try the older generateText endpoint as last resort
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateText?key={GEMINI_API_KEY}"
        payload = {
            "prompt": {
                "text": prompt
            },
            "temperature": 0.7,
            "maxOutputTokens": 500
        }
        
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'candidates' in data and len(data['candidates']) > 0:
                return data['candidates'][0]['output']
    except:
        pass
    
    logger.error(f"All Gemini API attempts failed. Last error: {last_error}")
    return f"Error: Could not connect to Gemini API. Please check your API key and try again. Details: {last_error}"

def extract_gemini_text(data):
    """Extract text from various Gemini response formats"""
    try:
        # Format 1: Standard response with candidates
        if 'candidates' in data and len(data['candidates']) > 0:
            candidate = data['candidates'][0]
            
            # Check different content formats
            if 'content' in candidate:
                content = candidate['content']
                if 'parts' in content and len(content['parts']) > 0:
                    if 'text' in content['parts'][0]:
                        return content['parts'][0]['text'].strip()
            
            # Direct output format
            if 'output' in candidate:
                return candidate['output'].strip()
        
        # Format 2: Alternative response structure
        if 'contents' in data and len(data['contents']) > 0:
            content = data['contents'][0]
            if 'parts' in content and len(content['parts']) > 0:
                if 'text' in content['parts'][0]:
                    return content['parts'][0]['text'].strip()
        
        # Format 3: Simple text response
        if 'text' in data:
            return data['text'].strip()
        
        # Format 4: GenerateText response
        if 'candidates' in data and len(data['candidates']) > 0:
            candidate = data['candidates'][0]
            if 'output' in candidate:
                return candidate['output'].strip()
        
        logger.error(f"Could not extract text from response: {data}")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting Gemini text: {str(e)}")
        return None
def get_gemini_response_sdk(message, conversation_history=[]):
    """Get response from Google Gemini API using the official SDK"""
    try:
        import google.generativeai as genai
        
        if not GEMINI_API_KEY:
            return "Error: Gemini API key not configured."
        
        # Configure the API
        genai.configure(api_key=GEMINI_API_KEY)
        
        # List available models (for debugging)
        try:
            models = genai.list_models()
            logger.info("Available models:")
            for model in models:
                logger.info(f"  - {model.name}")
        except:
            pass
        
        # Try different model names
        model_names = [
            'models/gemini-1.5-pro',
            'models/gemini-1.0-pro',
            'models/gemini-pro'
        ]
        
        last_error = None
        
        for model_name in model_names:
            try:
                # Create the model
                model = genai.GenerativeModel(model_name)
                
                # Build context from conversation history
                context = ""
                for entry in conversation_history[-3:]:
                    if entry.get('user'):
                        context += f"User: {entry['user']}\n"
                    if entry.get('bot'):
                        context += f"Assistant: {entry['bot']}\n"
                
                # Create prompt with history and current message
                prompt = f"{context}User: {message}\nAssistant:"
                
                # Generate response
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=500,
                        top_p=0.95,
                        top_k=40
                    )
                )
                
                if response.text:
                    logger.info(f"Successfully used Gemini model: {model_name}")
                    return response.text.strip()
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Failed with model {model_name}: {str(e)}")
                continue
        
        return f"Error: Could not generate response with any Gemini model. Last error: {last_error}"
        
    except ImportError:
        return "Error: google-generativeai package not installed. Run: pip install google-generativeai"
    except Exception as e:
        logger.error(f"Gemini SDK error: {str(e)}")
        return f"Error with Gemini SDK: {str(e)}"
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
        "gemini_configured": bool(GEMINI_API_KEY)
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
    elif API_PROVIDER == 'gemini' and not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set! Gemini API will not work.")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
