# setup.py - Run this script before starting the application
import nltk
import os
import json

def setup_environment():
    print("Setting up the Cyber Laws AI Assistant environment...")
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Download NLTK data
    print("Downloading NLTK data...")
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        print("NLTK data downloaded successfully.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        print("The application may still work with reduced functionality.")
    
    # Create basic knowledge base if it doesn't exist
    kb_path = 'data/cyber_laws_knowledge_base.json'
    if not os.path.exists(kb_path):
        print("Creating basic knowledge base...")
        basic_kb = {
            "laws": {
                "hacking": {
                    "description": "Unauthorized access to computer systems",
                    "keywords": ["hack", "unauthorized access", "system breach"],
                    "sections": {
                        "individual": "Section 66 of IT Act - Punishment for computer related offences",
                        "law_enforcement": "Section 66 IT Act, 2000 - Up to 3 years imprisonment or fine up to 5 lakh rupees or both"
                    },
                    "reporting_procedure": "File FIR at local police station or cyber crime portal",
                    "cases": ["State vs John Doe (2020)"]
                },
                "phishing": {
                    "description": "Fraudulent attempt to obtain sensitive information by disguising as a trustworthy entity",
                    "keywords": ["phish", "fake email", "identity fraud"],
                    "sections": {
                        "individual": "Sections 66C and 66D of IT Act - Identity theft and cheating by personation",
                        "law_enforcement": "Section 66C and 66D - Punishment up to 3 years and fine up to 1 lakh rupees"
                    },
                    "reporting_procedure": "Report to cyber crime portal and your bank if financial information was compromised",
                    "cases": ["Indian Bank vs Cyber Cell (2018)"]
                }
            },
            "intents": [
                {"tag": "greeting", "patterns": ["hi", "hello", "hey"], 
                 "responses": ["Hello! How can I help you with cyber law questions today?"]}
            ]
        }
        
        with open(kb_path, 'w') as file:
            json.dump(basic_kb, file, indent=4)
        print(f"Basic knowledge base created at {kb_path}")
    
    print("Setup complete! You can now run app.py to start the application.")

if __name__ == "__main__":
    setup_environment()