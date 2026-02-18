from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import uuid
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS for production
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost", "http://127.0.0.1", "http://your-domain.com", "*"],
        "methods": ["GET", POST, "OPTIONS", "DELETE"],
        "allow_headers": ["Content-Type"]
    }
})

# Store conversations (use Redis in production)
conversations = {}

# Get OpenAI key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set in environment variables")
    OPENAI_API_KEY = "your-key-here"  # Fallback, but better to fail

openai.api_key = OPENAI_API_KEY

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "AI Chatbot API is running in Docker",
        "version": "1.0.0",
        "container": True
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

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
        
        # Get AI response
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            ai_message = response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            ai_message = f"Error getting AI response: {str(e)}"
        
        # Store conversation
        conversations[conversation_id].append({
            "user": message,
            "bot": ai_message,
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify({
            "success": True,
            "message": ai_message,
            "conversation_id": conversation_id
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

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', '0') == '1'
    
    logger.info(f"Starting Flask app on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)
