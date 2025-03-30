# Simplified app.py with minimal dependencies
from flask import Flask, render_template, request, jsonify
import os
import json
import re

app = Flask(__name__)

class SimpleCyberLawAssistant:
    """Simplified assistant that doesn't rely on external NLP libraries"""
    def __init__(self):
        self.knowledge_base = self.load_knowledge_base('data/cyber_laws_knowledge_base.json')
        self.user_type = None
    
    def load_knowledge_base(self, filepath):
        """Load cyber laws knowledge base from JSON file with fallback"""
        try:
            with open(filepath, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Knowledge base file not found: {filepath}")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create basic structure
            basic_kb = {
                "laws": {
                    "hacking": {
                        "description": "Unauthorized access to computer systems",
                        "sections": {
                            "individual": "Section 66 of IT Act - Punishment for computer related offences",
                            "law_enforcement": "Section 66 IT Act, 2000 - Up to 3 years imprisonment or fine up to 5 lakh rupees or both"
                        },
                        "reporting_procedure": "File FIR at local police station or cyber crime portal",
                        "cases": ["State vs John Doe (2020)"]
                    }
                },
                "intents": [
                    {"tag": "greeting", "patterns": ["hi", "hello", "hey"], 
                     "responses": ["Hello! How can I help you with cyber law questions today?"]}
                ]
            }
            
            # Save the basic knowledge base
            with open(filepath, 'w') as file:
                json.dump(basic_kb, file, indent=4)
            
            return basic_kb
    
    def set_user_type(self, user_type):
        """Set whether the user is an individual or law enforcement"""
        if user_type.lower() in ["individual", "civilian", "person", "victim"]:
            self.user_type = "individual"
        elif user_type.lower() in ["police", "law enforcement", "officer", "department"]:
            self.user_type = "law_enforcement"
        else:
            self.user_type = None
    
    def process_user_input(self, user_input):
        """Simplified processing with pattern matching"""
        # Check if user is specifying their type
        if any(word in user_input.lower() for word in ["i am police", "law enforcement", "officer"]):
            self.set_user_type("law_enforcement")
            return "I understand you're with law enforcement. How can I help you with cyber law information today?"
        
        elif any(word in user_input.lower() for word in ["individual", "civilian", "victim"]):
            self.set_user_type("individual")
            return "I understand you're an individual seeking information. How can I help you with cyber law questions today?"
        
        # Check for greetings
        if any(word in user_input.lower() for word in ["hi", "hello", "hey", "good morning"]):
            return "Hello! How can I help you with cyber law questions today?"
        
        # Basic keyword matching for cybercrime types
        for crime_type, details in self.knowledge_base["laws"].items():
            if crime_type in user_input.lower() or any(keyword in user_input.lower() for keyword in details.get("keywords", [])):
                if "what is" in user_input.lower() or "definition" in user_input.lower():
                    return details.get("description", f"I don't have a definition for {crime_type}.")
                
                if "section" in user_input.lower() or "law" in user_input.lower():
                    section_info = details.get("sections", {})
                    if self.user_type:
                        return section_info.get(self.user_type, f"I don't have specific section information for {self.user_type}s regarding {crime_type}.")
                    return f"For {crime_type}, the relevant legal sections depend on whether you're an individual or law enforcement. Please specify your role."
                
                if "report" in user_input.lower() or "file" in user_input.lower() or "complain" in user_input.lower():
                    return details.get("reporting_procedure", f"I don't have reporting procedure information for {crime_type}.")
                
                # Default response for the crime type
                return f"{details.get('description', '')} This falls under {details.get('sections', {}).get('individual', 'certain sections of cyber law')}. Would you like to know more about reporting procedures or relevant case precedents?"
        
        # If no specific crime type is found
        return "I'm not sure which specific cybercrime you're asking about. I can provide information about hacking, phishing, online harassment, identity theft, and data breaches. Could you please be more specific?"

# Initialize the assistant
assistant = SimpleCyberLawAssistant()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    data = request.json
    user_input = data.get('query', '')
    
    if not user_input:
        return jsonify({'response': 'No query provided'})
    
    response = assistant.process_user_input(user_input)
    return jsonify({'response': response})

@app.route('/api/set_user_type', methods=['POST'])
def set_user_type():
    data = request.json
    user_type = data.get('user_type', '')
    
    if user_type in ['individual', 'law_enforcement']:
        assistant.set_user_type(user_type)
        return jsonify({'status': 'success', 'message': f'User type set to {user_type}'})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid user type'})

if __name__ == '__main__':
    app.run(debug=True)