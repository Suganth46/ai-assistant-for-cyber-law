
from flask import Flask, request, jsonify, render_template
from cyber_law_assistant import CyberLawAssistant
import logging

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Initialize the assistant
assistant = CyberLawAssistant(use_gpu=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_input = request.json.get('message', '')
        if not user_input:
            return jsonify({"response": "Please enter a valid question."}), 400
            
        response = assistant.process_input(user_input)
        return jsonify({"response": response})
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"response": "An error occurred while processing your request."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)