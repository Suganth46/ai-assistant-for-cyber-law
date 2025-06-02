# from cyber_law_assistant import CyberLawAssistant
from flask import Flask, render_template, request, jsonify
import requests
import json
import re

app = Flask(__name__)

# Initialize the Cyber Law Assistant
# assistant = CyberLawAssistant(use_gpu=False)

# Flowise API configuration
FLOWISE_API_URL = "http://localhost:3000/api/v1/prediction/627f1db3-aad6-49cf-b594-29f365c313e8"
API_KEY = "CscZGIeilk4Ucc2hTAbw4H4pqtJPDCIORexjwLiSxgY"
session_id = None
def call_flowise_api(user_input, session_id=None):
    """Call the Flowise API with the user input and session ID"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    if session_id == None:
        data = {
            "question": user_input
        }
    else:
        data = {
            "question": user_input,
            "overrideConfig": {
                "sessionId": session_id
            }
        }
    
    # Add session ID if provided
    
    
    try:
        response = requests.post(FLOWISE_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Flowise API: {e}")
        return None

def format_response(text):
    """Format the response text with paragraphs, newlines, and markdown-style bold"""
    if not text:
        return None
        
    # Split text into paragraphs (double newlines)
    paragraphs = text.split('\n\n')
    
    # Process each paragraph
    formatted_paragraphs = []
    for paragraph in paragraphs:
        # Convert single newlines to <br>
        paragraph = paragraph.replace('\n', '<br>')
        # Convert markdown-style bold to HTML bold
        paragraph = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', paragraph)
        # Add paragraph wrapper
        formatted_paragraphs.append(f'<p>{paragraph}</p>')
    
    return {
        'type': 'formatted_text',
        'content': ''.join(formatted_paragraphs)
    }

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/ask', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    
    
    # Get response from Flowise API with session ID
    flowise_response = call_flowise_api(user_input)
    
    if flowise_response:
        # Format the response
        response = format_response(flowise_response.get('text', ''))
        #session_id = flowise_response.get('sessionId', '')
        
    

        
        # Add session ID to response if provided by Flowise
        
    else:
        response = {
            'type': 'formatted_text',
            'content': '<p>Sorry, I could not process your request at this time.</p>'
        }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True) 