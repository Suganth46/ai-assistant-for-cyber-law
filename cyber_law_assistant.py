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

    def extract_query_intent(self, user_input):
        """Determine what type of information the user is seeking with improved accuracy"""
        query_lower = query.lower()

        # Use regex patterns for more precise matching
        patterns = {
            "definition": [r"what is", r"define", r"meaning", r"explain", r"describe"],
            "legal_sections": [r"section", r"law", r"act", r"legal", r"provision", r"statute"],
            "reporting": [r"report", r"file", r"complain", r"procedure", r"process", r"steps", r"how to"],
            "cases": [r"case", r"precedent", r"example", r"similar", r"court", r"ruling", r"judgment"],
            "prevention": [r"prevent", r"avoid", r"protect", r"secure", r"safeguard", r"precaution", r"safety"],
            "evidence": [r"evidence", r"proof", r"document", r"record", r"log", r"data", r"collect"],
            "punishment": [r"punish", r"penalty", r"sentence", r"fine", r"jail", r"prison", r"consequence"]
        }

        # Score each type based on pattern matches
        type_scores = {query_type: 0 for query_type in patterns}

        for query_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, query_lower):
                    type_scores[query_type] += 1

        # Find the highest scoring query type
        best_type = max(type_scores.items(), key=lambda x: x[1])

        # If we have a clear winner with at least one match
        if best_type[1] > 0:
            return best_type[0]

        # Fallback to general if no patterns match
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
        """Determine if the current query is a follow-up to the previous conversation with improved detection"""
        # Check for empty history
        if not self.conversation.conversation_history:
            return False

        # Get the most recent context
        last_law = self.conversation.context.get("last_law_category")
        current_topic = self.conversation.context.get("current_topic")

        # Basic pronoun and context indicator checks
        pronouns = ["it", "this", "that", "these", "those", "they", "them", "he", "she", "his", "her", "their"]
        follow_up_indicators = ["more", "also", "another", "additional", "further", "else", "again", "what about", "how about"]

        query_lower = query.lower()

        # Check for pronouns at the beginning of the query or embedded within
        for pronoun in pronouns:
            # At beginning
            if query_lower.startswith(pronoun + " "):
                return True
            # Within sentence but referring to context
            if " " + pronoun + " " in query_lower and last_law:
                # Check if the sentence is referring to the previous context
                return True

        # Check for follow-up indicators
        for indicator in follow_up_indicators:
            if indicator in query_lower:
                return True

        # Check if the query is very short (likely a follow-up)
        words = query_lower.split()
        if len(words) <= 3 and not any(greeting in query_lower for greeting in ["hi", "hello", "hey"]):
            return True
        # Check for implicit continuations (questions without clear subjects)
            question_starters = ["what", "how", "when", "where", "why", "who", "can", "should", "is", "are", "do", "does"]
        if any(query_lower.startswith(starter) for starter in question_starters):
            # If starts with question word and doesn't mention a new cybercrime category
            all_categories = list(self.knowledge_base.get("laws", {}).keys())
        if not any(category in query_lower for category in all_categories) and last_law:
            return True

        return False

    def handle_follow_up(self, query):
        """Handle follow-up questions by maintaining context with improved relevance"""
        last_law = self.conversation.context.get("last_law_category")
        current_topic = self.conversation.context.get("current_topic")
        # Get last few interactions for better context
        last_interactions = self.conversation.get_last_n_interactions(2)
        last_assistant_responses = [interaction.get("assistant_response", "") for interaction in last_interactions]
        last_user_inputs = [interaction.get("user_input", "") for interaction in last_interactions]

        if not last_law and not current_topic:
            return None, "I'm not sure what you're referring to. Could you provide more context or ask a more specific question?"

        query_type = self.extract_query_type(query)

        # Check if we're shifting query type within the same topic
        response = ""
        if query_type != "general" and last_law:
            base_response = self.get_law_info(last_law, query_type)

            # Add contextual transition based on previous exchanges
            response = f"Regarding {self.knowledge_base.get('laws', {}).get(last_law, {}).get('title', last_law)}, here's information about {query_type.replace('_', ' ')}:\n\n{base_response}"
        elif current_topic and current_topic.startswith("special_"):
            # Handle follow-ups for special categories
            special_info = self.knowledge_base.get("special_categories", {}).get(current_topic, {})
        if special_info:
            # Try to focus on the specific aspect being asked about
            if "provision" in query.lower() or "law" in query.lower():
                return current_topic, f"The special provisions for {current_topic.replace('_', ' ')} are: {special_info.get('special_provisions', 'Not specified')}"
            elif "report" in query.lower():
                return current_topic, f"Reporting requirements for {current_topic.replace('_', ' ')}: {special_info.get('reporting_requirements', special_info.get('reporting', 'Not specified'))}"
            else:
                # Generic response about the special category
                response = f"Regarding {current_topic.replace('_', ' ')}:\n" + "\n".join([f"• {key.replace('_', ' ')}: {value}" for key, value in special_info.items()])
        else:
            # If we can't determine the query type from the follow-up, provide general info
            if last_law:
                response = self.get_law_info(last_law, "general")

        return last_law or current_topic, response

    def answer_query(self, query):
        """Process the user query and generate a response with enhanced context handling"""
        try:
            # Check if it's a general intent
            intent = self.identify_intent(query)
            if intent:
                # Use conversation context to choose the most appropriate response
                if len(self.conversation.conversation_history) > 0:
                    # For returning users, use more personalized responses
                    response = intent.get("responses", ["I'm not sure how to respond to that."])[0]
                else:
                    # For new users, use general welcome responses
                    response = random.choice(intent.get("responses", ["I'm not sure how to respond to that."]))
                return response

            # Extract potential real-world scenario details
            scenario_details = self._extract_scenario_details(query)

            # Check if it's a follow-up question with enhanced detection
            if self.is_follow_up_question(query):
                category, response = self.handle_follow_up(query)

                # Enrich follow-up responses with context from previous interactions
                if category and len(self.conversation.get_last_n_interactions()) > 0:
                    context_summary = self.conversation.get_context_summary()
                    if context_summary:
                        response += f"\n\nBased on our conversation about {category}, I understand {context_summary.lower()}"

                return response

            # Identify law category with improved confidence scoring
            category = self.identify_law_category(query)
            confidence = self._calculate_category_confidence(query, category) if category else 0

            if category:
                # Update context with current law category
                self.conversation.update_context("last_law_category", category)
                self.conversation.update_context("current_topic", category)

                # Get query type
                query_type = self.extract_query_type(query)

                # Get specific information with scenario-specific details
                base_response = self.get_law_info(category, query_type)

                # Add real-world scenario adaptation if applicable
                if scenario_details and confidence > 0.6:
                    scenario_adaptation = self._adapt_to_scenario(category, query_type, scenario_details)
                    if scenario_adaptation:
                        base_response += f"\n\nIn your specific scenario: {scenario_adaptation}"

                return base_response

            # Check for multi-category questions
            categories = self._identify_multiple_categories(query)
            if len(categories) > 1:
                return self._generate_multi_category_response(categories, query)

            # If no category found, check special categories with improved matching
            special_match = self._check_special_categories(query)
            if special_match:
                return special_match

            # If still no match, provide a smart fallback with suggestions
            return self._generate_smart_fallback(query)

        except Exception as e:
            logger.error(f"Error in answer_query: {e}")
            return "I encountered an error processing your question. Could you try rephrasing it?"
    def _extract_scenario_details(self, query):
        """Extract real-world scenario details from the query"""
        scenario_details = {}

        # Look for temporal indicators
        time_patterns = [r'(\d+)\s+(day|week|month|year)s?\s+ago', r'yesterday', r'last\s+(week|month|year)']
        for pattern in time_patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                scenario_details['timeframe'] = matches[0] if isinstance(matches[0], str) else ' '.join(matches[0])

        # Look for action verbs indicating what happened
        action_verbs = ['hacked', 'stole', 'phished', 'breached', 'leaked', 'attacked', 'compromised', 'harassed']
        for verb in action_verbs:
            if verb in query.lower():
                scenario_details['action'] = verb

        # Look for affected targets/systems
        target_patterns = [r'my\s+(account|email|computer|phone|data|information)', r'our\s+(system|network|database|website)']
        for pattern in target_patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                scenario_details['target'] = matches[0]

        # Look for potential perpetrator information
        perpetrator_patterns = [r'someone', r'they', r'hacker', r'person', r'company', r'ex', r'stranger']
        for perp in perpetrator_patterns:
            if perp in query.lower().split():
                scenario_details['perpetrator'] = perp

        return scenario_details

def _adapt_to_scenario(self, category, query_type, scenario_details):
    """Adapt the response to the specific scenario details provided"""
    adaptations = []

    # Tailor response based on the specific action that occurred
    if 'action' in scenario_details:
        action = scenario_details['action']
        if query_type == "reporting":
            adaptations.append(f"Since your {scenario_details.get('target', 'system')} was {action}, be sure to include exact details of how the {action} occurred in your report.")
        elif query_type == "evidence":
            adaptations.append(f"For a {action} incident, focus on collecting {self._get_priority_evidence(action)} as your top priority evidence.")

    # Tailor response based on timeframe
    if 'timeframe' in scenario_details:
        timeframe = scenario_details['timeframe']
        if 'month' in timeframe or 'year' in timeframe:
            adaptations.append(f"Since this happened {timeframe}, focus on gathering any archived logs or backups that might still contain evidence.")
        elif 'day' in timeframe or 'week' in timeframe or 'yesterday' in timeframe:
            adaptations.append(f"This is a recent incident ({timeframe}), so prioritize preserving all current digital evidence before it's overwritten.")

    return " ".join(adaptations) if adaptations else ""

def _get_priority_evidence(self, action):
    """Get priority evidence types based on the specific action"""
    evidence_map = {
        'hacked': "system logs, unusual login attempts, and modified files",
        'phished': "the phishing email with full headers and any websites you were directed to",
        'breached': "access logs, data access records, and any unauthorized account activities",
        'leaked': "copies of the leaked information and where it appeared online",
        'harassed': "screenshots of all communications with timestamps and account information"
    }

    return evidence_map.get(action, "all digital communications and system activities")

def _identify_multiple_categories(self, query):
    """Identify if query spans multiple cybercrime categories"""
    categories = []
    category_confidence = {}

    # Check each category keyword against the query
    for category, details in self.knowledge_base.get("laws", {}).items():
        # Check title match
        if category in query.lower() or details.get('title', '').lower() in query.lower():
            categories.append(category)
            category_confidence[category] = 0.9
            continue

        # Check keywords
        hit_count = 0
        for keyword in details.get('keywords', []):
            if keyword.lower() in query.lower():
                hit_count += 1

        if hit_count >= 1:
            confidence = min(0.5 + (hit_count * 0.1), 0.9)  # Cap at 0.9
            categories.append(category)
            category_confidence[category] = confidence

    # Sort by confidence
    return sorted(categories, key=lambda c: category_confidence.get(c, 0), reverse=True)

def _generate_multi_category_response(self, categories, query):
    """Generate a response that addresses multiple cybercrime categories"""
    if len(categories) <= 1:
        return self.get_law_info(categories[0], "general")

    # Determine what the user is asking about these multiple categories
    if "difference" in query.lower() or "versus" in query.lower() or " vs " in query.lower():
        return self._explain_differences(categories[:2])  # Focus on top 2 for clarity

    # If asking which law applies
    if "which law" in query.lower() or "what law" in query.lower() or "legal" in query.lower():
        return self._summarize_relevant_laws(categories[:3])  # Top 3 categories

    # Default multi-category response
    response = "Your question appears to involve multiple cyber law areas:\n\n"
    for category in categories[:3]:  # Limit to top 3 for clarity
        law_info = self.knowledge_base.get("laws", {}).get(category, {})
        response += f"• {law_info.get('title', category.capitalize())}: {law_info.get('description', 'No description available.')}\n"

    response += "\nFor more specific information, could you clarify which aspect you're most interested in?"
    return response

def _explain_differences(self, categories):
    """Explain the differences between two cybercrime categories"""
    if len(categories) < 2:
        return self.get_law_info(categories[0], "general")

    cat1 = self.knowledge_base.get("laws", {}).get(categories[0], {})
    cat2 = self.knowledge_base.get("laws", {}).get(categories[1], {})

    response = f"Comparing {cat1.get('title', categories[0])} and {cat2.get('title', categories[1])}:\n\n"

    # Definition differences
    response += "Definitions:\n"
    response += f"• {cat1.get('title')}: {cat1.get('description')}\n"
    response += f"• {cat2.get('title')}: {cat2.get('description')}\n\n"

    # Legal framework differences
    response += "Legal Framework:\n"
    response += f"• {cat1.get('title')}: {', '.join(cat1.get('legal_framework', {}).get('primary_sections', ['N/A']))}\n"
    response += f"• {cat2.get('title')}: {', '.join(cat2.get('legal_framework', {}).get('primary_sections', ['N/A']))}\n\n"

    # Punishment differences
    response += "Punishment:\n"
    response += f"• {cat1.get('title')}: {cat1.get('legal_framework', {}).get('punishment', 'Not specified')}\n"
    response += f"• {cat2.get('title')}: {cat2.get('legal_framework', {}).get('punishment', 'Not specified')}\n"

    return response

def _summarize_relevant_laws(self, categories):
    """Summarize relevant legal sections for multiple categories"""
    response = "Relevant legal sections for your query:\n\n"

    for category in categories:
        law_info = self.knowledge_base.get("laws", {}).get(category, {})
        legal_framework = law_info.get("legal_framework", {})

        response += f"For {law_info.get('title', category)}:\n"
        response += f"• Primary sections: {', '.join(legal_framework.get('primary_sections', ['Not specified']))}\n"
        response += f"• Punishment: {legal_framework.get('punishment', 'Not specified')}\n\n"

    return response

def _check_special_categories(self, query):
    """Check for special categories with improved detection"""
    special_categories = self.knowledge_base.get("special_categories", {})

    # First check direct matches
    for category_name, category_info in special_categories.items():
        category_display = category_name.replace("_", " ")
        if category_display in query.lower():
            self.conversation.update_context("current_topic", category_name)
            return f"Information about {category_display}:\n" + "\n".join([f"• {key.replace('_', ' ')}: {value}" for key, value in category_info.items()])

    # Then check indirect references
    for category_name, category_info in special_categories.items():
        # Check for category-specific keywords
        category_keywords = []
        if category_name == "children_protection":
            category_keywords = ["minor", "child", "kid", "underage", "young", "youth"]
        elif category_name == "critical_infrastructure":
            category_keywords = ["infrastructure", "critical", "essential service", "utility", "national security"]

        for keyword in category_keywords:
            if keyword in query.lower():
                category_display = category_name.replace("_", " ")
                self.conversation.update_context("current_topic", category_name)
                return f"Since your query involves {keyword}, here's information about {category_display}:\n" + "\n".join([f"• {key.replace('_', ' ')}: {value}" for key, value in category_info.items()])

    return None

def _generate_smart_fallback(self, query):
    """Generate a smart fallback response with helpful suggestions"""
    # Extract key terms from the query
    query_terms = set(self.preprocess_text(query).split())

    # Find the most relevant categories based on term overlap
    category_relevance = {}
    for category, details in self.knowledge_base.get("laws", {}).items():
        # Create a set of words from title, description and keywords
        category_words = set(self.preprocess_text(details.get("title", "")).split())
        category_words.update(self.preprocess_text(details.get("description", "")).split())
        for keyword in details.get("keywords", []):
            category_words.update(self.preprocess_text(keyword).split())

        # Calculate overlap
        overlap = len(query_terms.intersection(category_words))
        if overlap > 0:
            category_relevance[category] = overlap

    # If we found some potentially relevant categories
    if category_relevance:
        top_categories = sorted(category_relevance.items(), key=lambda x: x[1], reverse=True)[:3]

        response = "I'm not sure exactly what cyber law information you're seeking. You might be interested in:"
        for category, _ in top_categories:
            law_info = self.knowledge_base.get("laws", {}).get(category, {})
            response += f"\n• {law_info.get('title', category.capitalize())}"

        response += "\n\nCould you specify which of these areas you'd like information about, or clarify your question?"
        return response
    def _calculate_category_confidence(self, query, category):
        """Calculate confidence score for category identification"""
        if not category:
            return 0

        # Get category info
        cat_info = self.knowledge_base.get("laws", {}).get(category, {})
        if not cat_info:
            return 0.3  # Base confidence if we have the category but no details

        # Start with base confidence
        confidence = 0.5

        # Direct category name match is high confidence
        if category in query.lower():
            confidence += 0.3

        # Check for title match
        if cat_info.get("title", "").lower() in query.lower():
            confidence += 0.3

        # Count keyword matches
        keyword_matches = 0
        for keyword in cat_info.get("keywords", []):
            if keyword.lower() in query.lower():
                keyword_matches += 1

        # Add confidence based on keyword matches (diminishing returns)
        if keyword_matches > 0:
            confidence += min(0.4, 0.1 * keyword_matches)

        # Cap at 1.0
        return min(confidence, 1.0)
        # Generic fallback
        return "I'm not sure which cyber law you're asking about. Could you provide more details or specify the type of cybercrime (e.g., hacking, phishing, online harassment, identity theft) you're interested in?"
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
