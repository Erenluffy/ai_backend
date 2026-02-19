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
API_PROVIDER = os.getenv('API_PROVIDER', 'deepseek').lower()  # deepseek, openai, anthropic, gemini
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
    """Get response from DeepSeek API with SmartMulch context"""
    if not DEEPSEEK_API_KEY:
        return "I'm sorry, but the AI service is not properly configured. Please contact support."
    
    # Get relevant context from knowledge base
    context = knowledge_base.get_context_for_query(message)
    
    # Create system prompt with SmartMulch context
    system_prompt = f"""You are SmartMulch AI Assistant, a helpful agricultural expert specializing in smart farming, mulching techniques, and sustainable agriculture.

You have access to the following information about SmartMulch and agriculture:

{context}

Guidelines:
1. Always be helpful, friendly, and professional
2. Focus on agriculture, farming, and SmartMulch products
3. If asked about topics outside agriculture, politely redirect to farming topics
4. Provide practical, actionable advice for farmers
5. Mention SmartMulch products when relevant, but don't force it
6. Be encouraging and supportive of sustainable farming practices
7. Never mention that you're using an AI API or that you have limitations
8. Respond as if you're a knowledgeable agricultural consultant

Remember: You are completely free to use and always here to help with farming questions!"""
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Prepare messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    for entry in conversation_history[-5:]:
        if entry.get('user'):
            messages.append({"role": "user", "content": entry['user']})
        if entry.get('bot'):
            messages.append({"role": "assistant", "content": entry['bot']})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.4,
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
            return "I'm here to help with your farming questions! Could you please rephrase that?"
            
    except requests.exceptions.RequestException as e:
        logger.error(f"DeepSeek API error: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Response: {e.response.text}")
        return "I'm having trouble connecting right now. Please try again in a moment."


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
        "message": "SmartMulch AI Assistant is ready to help!",
        "version": "2.0.0",
        "features": [
            "Farming recommendations",
            "Crop guides",
            "Soil analysis",
            "Smart mulching advice",
            "Sustainable farming tips"
        ]
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "service": "SmartMulch AI Assistant"
    }), 200

@app.route('/api/knowledge/search', methods=['POST'])
def search_knowledge():
    """Search the knowledge base directly"""
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    results = knowledge_base.search_knowledge(query)
    return jsonify({
        "success": True,
        "results": results
    })

@app.route('/api/knowledge/topics', methods=['GET'])
def get_topics():
    """Get all main topics in knowledge base"""
    topics = list(knowledge_base.knowledge.keys())
    return jsonify({
        "success": True,
        "topics": topics
    })
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
        
        # Get conversation history
        history = conversations[conversation_id]
        
        # Get AI response with SmartMulch context
        ai_message = get_deepseek_response(message, history)
        
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
            "assistant": "SmartMulch AI"
        })
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "I'm having trouble processing your request. Please try again."
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
