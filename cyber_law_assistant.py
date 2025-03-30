# Cyber Laws AI Assistant
# An NLP-powered application to provide guidance on cyber laws with conversation history

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import pickle
import re
import json
import os
import random
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(
    filename='cyber_law_assistant.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation history and context for improved responses"""
    
    def __init__(self, max_history=10):
        self.conversation_history = []
        self.max_history = max_history
        self.context = {
            "user_type": None,
            "current_topic": None,
            "last_law_category": None,
            "user_info": {},
            "session_id": self._generate_session_id()
        }
    
    def _generate_session_id(self):
        """Generate a unique session ID"""
        return f"session_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
    
    def add_interaction(self, user_input, assistant_response):
        """Add an interaction to the conversation history"""
        timestamp = datetime.datetime.now().isoformat()
        
        interaction = {
            "timestamp": timestamp,
            "user_input": user_input,
            "assistant_response": assistant_response,
            "context": self.context.copy()
        }
        
        self.conversation_history.append(interaction)
        
        # Keep history within max size
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        # Log the interaction
        logger.info(f"Session {self.context['session_id']} - User: {user_input[:50]}...")
        logger.info(f"Session {self.context['session_id']} - Assistant: {assistant_response[:50]}...")
    
    def get_last_n_interactions(self, n=3):
        """Get the last n interactions for context"""
        return self.conversation_history[-n:] if len(self.conversation_history) >= n else self.conversation_history
    
    def update_context(self, key, value):
        """Update a specific context value"""
        self.context[key] = value
        logger.debug(f"Updated context: {key}={value}")
    
    def get_context_summary(self):
        """Get a text summary of current context for better responses"""
        summary = []
        
        if self.context["user_type"]:
            summary.append(f"User is {self.context['user_type']}.")
        
        if self.context["current_topic"]:
            summary.append(f"Current topic is {self.context['current_topic']}.")
        
        if self.context["last_law_category"]:
            summary.append(f"Last discussed law category was {self.context['last_law_category']}.")
        
        return " ".join(summary)
    
    def save_conversation(self, filepath):
        """Save the conversation history to a file"""
        try:
            with open(filepath, 'w') as file:
                json.dump({
                    "session_id": self.context["session_id"],
                    "timestamp": datetime.datetime.now().isoformat(),
                    "conversation": self.conversation_history
                }, file, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False
    
    def load_conversation(self, filepath):
        """Load a conversation history from a file"""
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
                self.conversation_history = data.get("conversation", [])
                self.context["session_id"] = data.get("session_id", self._generate_session_id())
            return True
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            return False


class CyberLawAssistant:
    def __init__(self, knowledge_base_path='data/cyber_laws_knowledge_base.json'):
        """Initialize the Cyber Law Assistant with enhanced error handling"""
        try:
            # Initialize NLTK components
            self.initialize_nltk()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Load the knowledge base
            self.knowledge_base_path = knowledge_base_path
            self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
            
            # Initialize TF-IDF vectorizer for intent matching
            self.vectorizer = TfidfVectorizer()
            self.fit_vectorizer()
            
            # Initialize conversation manager
            self.conversation = ConversationManager()
            
            logger.info("CyberLawAssistant initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing CyberLawAssistant: {e}")
            # Still initialize the conversation manager even if other components fail
            self.conversation = ConversationManager()
            print(f"Error during initialization: {e}")
    
    def initialize_nltk(self):
        """Download required NLTK resources with error handling"""
        try:
            # Download essential NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logger.warning(f"Error downloading NLTK data: {e}")
            logger.warning("Using fallback mechanisms")
    
    def preprocess_text(self, text):
        """Preprocess the input text with robust error handling"""
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text input: {type(text)}")
            return ""
            
        # Convert to lowercase and remove special characters
        try:
            text = re.sub(r'[^\w\s]', '', text.lower())
        except:
            text = text.lower()
        
        # Tokenize with error handling
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback to simple whitespace tokenization if NLTK fails
            tokens = text.split()
            logger.debug("Using simple tokenization as fallback")
        
        # Remove stopwords and lemmatize
        try:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        except Exception:
            # Skip lemmatization if not available
            tokens = [word for word in tokens if word not in self.stop_words]
            logger.debug("Skipping lemmatization due to resource unavailability")
        
        return " ".join(tokens)
        
    def load_knowledge_base(self, filepath):
        """Load cyber laws knowledge base from JSON file with robust error handling"""
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                logger.warning(f"Knowledge base file not found: {filepath}")
                # Try to create directory if needed
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Return a basic structure
                return self._create_basic_knowledge_base()
            
            # Load the file
            with open(filepath, 'r') as file:
                data = json.load(file)
                logger.info(f"Knowledge base loaded successfully from {filepath}")
                return data
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in knowledge base: {filepath}")
            return self._create_basic_knowledge_base()
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return self._create_basic_knowledge_base()
    
    def _create_basic_knowledge_base(self):
        """Create a basic knowledge base structure"""
        logger.info("Creating basic knowledge base structure")
        return {
            "laws": {
                "hacking": {
                    "title": "Computer Hacking",
                    "description": "Unauthorized access to computer systems",
                    "keywords": ["hack", "unauthorized access", "breach"],
                    "legal_framework": {
                        "primary_sections": ["Section 66 of IT Act, 2000"],
                        "punishment": "Up to 3 years imprisonment or fine up to 5 lakh rupees or both"
                    },
                    "reporting_procedure": {
                        "individual": ["File FIR at local police station or cyber crime portal"]
                    },
                    "evidence_collection": ["Server logs", "Access records"],
                    "landmark_cases": [
                        {
                            "name": "State vs John Doe (2020)",
                            "significance": "Example case"
                        }
                    ]
                }
            },
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["hi", "hello", "hey"],
                    "responses": ["Hello! How can I help you with cyber law questions today?"]
                },
                {
                    "tag": "goodbye",
                    "patterns": ["bye", "thanks", "thank you"],
                    "responses": ["Goodbye! Feel free to return if you have more questions about cyber laws."]
                }
            ]
        }
    
    def fit_vectorizer(self):
        """Prepare the TF-IDF vectorizer with intent patterns"""
        try:
            patterns = []
            for intent in self.knowledge_base.get("intents", []):
                patterns.extend(intent.get("patterns", []))
            
            # Ensure we have patterns before fitting
            if not patterns:
                patterns = ["default pattern"]
                
            processed_patterns = [self.preprocess_text(pattern) for pattern in patterns]
            self.vectorizer.fit(processed_patterns)
            logger.debug("TF-IDF vectorizer fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting vectorizer: {e}")
            # Initialize with a basic fit
            self.vectorizer.fit(["hello", "goodbye", "help"])
    
    def identify_intent(self, query):
        """Identify the intent of user query with improved error handling"""
        try:
            processed_query = self.preprocess_text(query)
            
            if not processed_query:
                return None
            
            # Transform the query using the fitted vectorizer
            query_vector = self.vectorizer.transform([processed_query])
            
            best_match = None
            highest_similarity = -1
            
            for intent in self.knowledge_base.get("intents", []):
                for pattern in intent.get("patterns", []):
                    processed_pattern = self.preprocess_text(pattern)
                    pattern_vector = self.vectorizer.transform([processed_pattern])
                    
                    similarity = cosine_similarity(query_vector, pattern_vector)[0][0]
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = intent
            
            # Threshold for intent matching
            if highest_similarity > 0.4:
                return best_match
            return None
        except Exception as e:
            logger.error(f"Error in intent identification: {e}")
            return None
    
    def identify_law_category(self, query):
        """Identify which cyber law category the query is about with improved matching"""
        try:
            processed_query = self.preprocess_text(query)
            
            if not processed_query:
                return None
                
            best_category = None
            highest_match_count = 0
            
            # Split the processed query into words for better matching
            query_words = set(processed_query.split())
            
            for category, details in self.knowledge_base.get("laws", {}).items():
                # Search for category name in query
                if category in processed_query:
                    return category
                
                # Count matching keywords for better matching
                match_count = 0
                
                # Check title words
                title_words = set(self.preprocess_text(details.get("title", "")).split())
                match_count += len(query_words.intersection(title_words))
                
                # Check description words
                desc_words = set(self.preprocess_text(details.get("description", "")).split())
                match_count += len(query_words.intersection(desc_words))
                
                # Search for keywords related to the category
                for keyword in details.get("keywords", []):
                    keyword_processed = self.preprocess_text(keyword)
                    if keyword_processed in processed_query:
                        match_count += 2  # Give higher weight to keyword matches
                
                if match_count > highest_match_count:
                    highest_match_count = match_count
                    best_category = category
            
            # Only return if we have enough matches
            if highest_match_count >= 1:
                return best_category
                
            return None
        except Exception as e:
            logger.error(f"Error in law category identification: {e}")
            return None
    
    def set_user_type(self, user_type):
        """Set whether the user is an individual or law enforcement"""
        if user_type.lower() in ["individual", "civilian", "person", "victim"]:
            self.conversation.update_context("user_type", "individual")
        elif user_type.lower() in ["police", "law enforcement", "officer", "department"]:
            self.conversation.update_context("user_type", "law_enforcement")
        else:
            self.conversation.update_context("user_type", "unspecified")
    
    def extract_query_type(self, query):
        """Determine what type of information the user is seeking"""
        query_lower = query.lower()
        
        if "what is" in query_lower or "define" in query_lower or "meaning" in query_lower:
            return "definition"
        elif "section" in query_lower or "law" in query_lower or "act" in query_lower or "legal" in query_lower:
            return "legal_sections"
        elif "report" in query_lower or "file" in query_lower or "complain" in query_lower or "procedure" in query_lower:
            return "reporting"
        elif "case" in query_lower or "precedent" in query_lower or "example" in query_lower:
            return "cases"
        elif "prevent" in query_lower or "avoid" in query_lower or "protect" in query_lower:
            return "prevention"
        elif "evidence" in query_lower or "proof" in query_lower or "document" in query_lower:
            return "evidence"
        elif "punishment" in query_lower or "penalty" in query_lower or "sentence" in query_lower or "fine" in query_lower:
            return "punishment"
        else:
            return "general"
    
    def get_law_info(self, category, query_type):
        """Get specific information from a law category based on query type"""
        try:
            law_info = self.knowledge_base.get("laws", {}).get(category, {})
            
            if not law_info:
                return f"I don't have information about {category} in my knowledge base."
            
            user_type = self.conversation.context.get("user_type", "individual")
            
            if query_type == "definition":
                return f"{law_info.get('title', category.capitalize())}: {law_info.get('description', 'No description available.')}"
            
            elif query_type == "legal_sections":
                legal_framework = law_info.get("legal_framework", {})
                primary_sections = legal_framework.get("primary_sections", ["No specific sections available"])
                
                if user_type == "law_enforcement":
                    return f"Legal framework for {law_info.get('title', category)}: {', '.join(primary_sections)}. Punishment: {legal_framework.get('punishment', 'Not specified')}."
                else:
                    relevant_sections = legal_framework.get("relevant_sections", {})
                    individual_section = relevant_sections.get("individual_victims", "No specific section mentioned")
                    return f"For {law_info.get('title', category)}, the relevant legal section is: {individual_section}."
            
            elif query_type == "reporting":
                procedures = law_info.get("reporting_procedure", {})
                
                if user_type == "law_enforcement":
                    steps = procedures.get("law_enforcement", procedures.get("corporate", ["No specific procedure available"]))
                else:
                    steps = procedures.get("individual", ["No specific procedure available"])
                
                return f"Reporting procedure for {law_info.get('title', category)}:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
            
            elif query_type == "cases":
                cases = law_info.get("landmark_cases", [])
                if not cases:
                    return f"I don't have case precedent information for {law_info.get('title', category)}."
                
                case_list = "\n".join([f"• {case.get('name', 'Unnamed case')}: {case.get('significance', 'No details available')}" for case in cases])
                return f"Landmark cases for {law_info.get('title', category)}:\n{case_list}"
            
            elif query_type == "prevention":
                tips = law_info.get("prevention_tips", ["No specific prevention tips available"])
                return f"Prevention tips for {law_info.get('title', category)}:\n" + "\n".join([f"{i+1}. {tip}" for i, tip in enumerate(tips)])
            
            elif query_type == "evidence":
                evidence = law_info.get("evidence_collection", ["No specific evidence collection guidelines available"])
                return f"Evidence collection for {law_info.get('title', category)}:\n" + "\n".join([f"{i+1}. {item}" for i, item in enumerate(evidence)])
            
            elif query_type == "punishment":
                legal_framework = law_info.get("legal_framework", {})
                return f"Punishment for {law_info.get('title', category)}: {legal_framework.get('punishment', 'Not specified in my knowledge base')}"
            
            else:
                # For general queries, provide a comprehensive overview
                title = law_info.get('title', category.capitalize())
                description = law_info.get('description', 'No description available')
                framework = law_info.get('legal_framework', {}).get('primary_sections', ['No specific sections available'])
                
                return f"{title}: {description}\n\nKey legal sections: {', '.join(framework)}\n\nUse more specific queries about reporting, evidence, prevention, or cases for detailed information."
                
        except Exception as e:
            logger.error(f"Error retrieving law info: {e}")
            return f"I encountered an error while retrieving information about {category}. Please try a different question."
    
    def get_similarity_score(self, text1, text2):
        """Get similarity score between two texts"""
        try:
            # Preprocess both texts
            processed_text1 = self.preprocess_text(text1)
            processed_text2 = self.preprocess_text(text2)
            
            # Vectorize
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
            
            # Calculate similarity
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0
    
    def is_follow_up_question(self, query):
        """Determine if the current query is a follow-up to the previous conversation"""
        # Check for pronouns and context indicators
        pronouns = ["it", "this", "that", "these", "those", "they", "them"]
        follow_up_indicators = ["more", "also", "another", "additional", "further", "else", "again"]
        
        query_lower = query.lower()
        
        # Check for pronouns at the beginning of the query
        for pronoun in pronouns:
            if query_lower.startswith(pronoun + " ") or " " + pronoun + " " in query_lower:
                return True
        
        # Check for follow-up indicators
        for indicator in follow_up_indicators:
            if indicator in query_lower:
                return True
        
        # Check if the query is very short (likely a follow-up)
        if len(query_lower.split()) <= 3 and not any(greeting in query_lower for greeting in ["hi", "hello", "hey"]):
            return True
            
        return False
    
    def handle_follow_up(self, query):
        """Handle follow-up questions by maintaining context"""
        last_law = self.conversation.context.get("last_law_category")
        
        if not last_law:
            return None, "I'm not sure what you're referring to. Could you provide more context or ask a more specific question?"
        
        query_type = self.extract_query_type(query)
        return last_law, self.get_law_info(last_law, query_type)
    
    def answer_query(self, query):
        """Process the user query and generate a response"""
        try:
            # Check if it's a general intent
            intent = self.identify_intent(query)
            if intent:
                response = random.choice(intent.get("responses", ["I'm not sure how to respond to that."]))
                return response
            
            # Check if it's a follow-up question
            if self.is_follow_up_question(query):
                category, response = self.handle_follow_up(query)
                return response
            
            # Identify law category
            category = self.identify_law_category(query)
            
            if category:
                # Update context with current law category
                self.conversation.update_context("last_law_category", category)
                self.conversation.update_context("current_topic", category)
                
                # Get query type
                query_type = self.extract_query_type(query)
                
                # Get specific information
                return self.get_law_info(category, query_type)
            
            # If we couldn't identify a category, check special categories
            special_categories = self.knowledge_base.get("special_categories", {})
            for category_name, category_info in special_categories.items():
                category_keywords = [category_name.replace("_", " ")] + list(category_info.keys())
                for keyword in category_keywords:
                    if keyword in query.lower():
                        # Found a match in special categories
                        self.conversation.update_context("current_topic", category_name)
                        return f"Information about {category_name.replace('_', ' ')}:\n" + "\n".join([f"• {key.replace('_', ' ')}: {value}" for key, value in category_info.items()])
            
            # If no category found, provide a general response
            return "I'm not sure which cyber law you're asking about. Could you provide more details or specify the type of cybercrime (e.g., hacking, phishing, online harassment)?"
            
        except Exception as e:
            logger.error(f"Error in answer_query: {e}")
            return "I encountered an error processing your question. Could you try rephrasing it?"
    
    def process_user_input(self, user_input):
        """Main function to process user input and return appropriate response"""
        try:
            # Check for empty or invalid input
            if not user_input or not isinstance(user_input, str) or user_input.strip() == "":
                return "I didn't catch that. Could you please provide a question about cyber laws?"
            
            # Check if user is specifying their type
            if "i am" in user_input.lower() or "i'm a" in user_input.lower() or "i'm an" in user_input.lower():
                if any(term in user_input.lower() for term in ["police", "officer", "law enforcement"]):
                    self.set_user_type("law_enforcement")
                    response = "I understand you're with law enforcement. How can I help you with cyber law information today?"
                    self.conversation.add_interaction(user_input, response)
                    return response
                elif any(term in user_input.lower() for term in ["individual", "victim", "civilian", "person"]):
                    self.set_user_type("individual")
                    response = "I understand you're an individual seeking information. How can I help you with cyber law questions today?"
                    self.conversation.add_interaction(user_input, response)
                    return response
            
            # Process the actual query
            response = self.answer_query(user_input)
            
            # Add to conversation history
            self.conversation.add_interaction(user_input, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in process_user_input: {e}")
            return "I encountered an unexpected error. Please try asking your question again."
    
    def save_conversation_history(self, filepath=None):
        """Save the current conversation history to a file"""
        if not filepath:
            # Generate a timestamped filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"conversations/conversation_{timestamp}.json"
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        return self.conversation.save_conversation(filepath)
    
    def load_conversation_history(self, filepath):
        """Load conversation history from a file"""
        return self.conversation.load_conversation(filepath)
    
    def update_knowledge_base(self, new_data):
        """Update the knowledge base with new information"""
        try:
            # Update in-memory knowledge base
            for key, value in new_data.items():
                if key in self.knowledge_base and isinstance(self.knowledge_base[key], dict) and isinstance(value, dict):
                    self.knowledge_base[key].update(value)
                else:
                    self.knowledge_base[key] = value
            
            # Save to file
            with open(self.knowledge_base_path, 'w') as file:
                json.dump(self.knowledge_base, file, indent=4)
            
            # Re-fit the vectorizer with new data
            self.fit_vectorizer()
            
            logger.info("Knowledge base updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating knowledge base: {e}")
            return False


# Example usage with CLI interface
def run_cli():
    # Create an instance of the assistant
    assistant = CyberLawAssistant()
    
    print("\n===== Cyber Law AI Assistant =====")
    print("Type 'exit' to end the conversation")
    print("Type 'save' to save the conversation")
    print("Type 'help' for available commands")
    print("==================================\n")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            print("\nAssistant: Thank you for using the Cyber Law Assistant. Goodbye!")
            # Save conversation on exit
            assistant.save_conversation_history()
            break
        
        elif user_input.lower() == 'save':
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversations/conversation_{timestamp}.json"
            if assistant.save_conversation_history(filename):
                print(f"\nAssistant: Conversation saved to {filename}")
            else:
                print("\nAssistant: Failed to save conversation.")
            continue
        
        elif user_input.lower() == 'help':
            print("\nAssistant: Available commands:")
            print("  'exit' - End the conversation")
            print("  'save' - Save the current conversation")
            print("  'help' - Show this help message")
            print("  'clear' - Clear the conversation history")
            print("  'status' - Show the current conversation status")
            continue
        
        elif user_input.lower() == 'clear':
            assistant.conversation = ConversationManager()
            print("\nAssistant: Conversation history cleared.")
            continue
        
        elif user_input.lower() == 'status':
            context = assistant.conversation.context
            print("\nAssistant: Current conversation status:")
            print(f"  Session ID: {context.get('session_id', 'None')}")
            print(f"  User type: {context.get('user_type', 'Not specified')}")
            print(f"  Current topic: {context.get('current_topic', 'None')}")
            print(f"  Conversation length: {len(assistant.conversation.conversation_history)} interactions")
            continue
        
        response = assistant.process_user_input(user_input)
        print(f"\nAssistant: {response}")


# For importable module
if __name__ == "__main__":
    run_cli()