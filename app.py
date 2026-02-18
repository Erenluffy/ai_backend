from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import openai
import uuid
from datetime import datetime
import os

app = Flask(__name__, static_folder='static', static_url_path='')
# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Store conversations
conversations = {}

# Your OpenAI API key
OPENAI_API_KEY = "ravi"
openai.api_key = OPENAI_API_KEY

def get_ai_response(message, conversation_history=None):
    """Get response from OpenAI"""
    try:
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        
        if conversation_history:
            for msg in conversation_history[-5:]:
                messages.append({"role": "user", "content": msg["user"]})
                messages.append({"role": "assistant", "content": msg["bot"]})
        
        messages.append({"role": "user", "content": message})
        
        # Using OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "AI Chatbot API is running",
        "server_ip": "178.128.179.242",
        "port": 5000,
        "version": "1.0.0"
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
        
        # Get AI response
        ai_message = get_ai_response(message, conversations[conversation_id])
        
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
        print(f"Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"status": "ok", "message": "Backend is working!"})

@app.route('/api/history/<conversation_id>', methods=['GET'])
def get_history(conversation_id):
    if conversation_id in conversations:
        return jsonify({"success": True, "history": conversations[conversation_id]})
    return jsonify({"success": False, "error": "Not found"}), 404

@app.route('/api/clear/<conversation_id>', methods=['DELETE', 'OPTIONS'])
def clear_history(conversation_id):
    if request.method == 'OPTIONS':
        return '', 200
    if conversation_id in conversations:
        conversations[conversation_id] = []
        return jsonify({"success": True, "message": "Cleared"})
    return jsonify({"success": False, "error": "Not found"}), 404

if __name__ == '__main__':
    print("\n" + "="*60)
    print("✅ AI CHATBOT BACKEND - RUNNING")
    print("="*60)
    print(f"📍 Local URL: http://127.0.0.1:5000")
    print(f"📍 Network URL: http://178.128.179.242:5000")
    print(f"📍 Test endpoint: http://178.128.179.242:5000/api/test")
    print("="*60)
    print("🚀 Server is ready for connections!")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
