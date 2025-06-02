from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import json
import datetime
import logging
import requests
import time
import os
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from auth import User, init_db, login_manager, register_user, authenticate_user
from auth import save_conversation, get_user_conversations, get_conversation_messages, save_feedback
from pymongo import MongoClient
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import re
from functools import lru_cache
import hashlib
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
import google.oauth2.id_token
from google.auth.transport import requests as google_requests
from analytics_dashboard import create_dashboard

# Load environment variables
load_dotenv()

# Flask configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'cyberlaw-assistant-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=7)

# Set up logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.getenv('LOG_FILE', 'app.log')
)
logger = logging.getLogger(__name__)

# Google OAuth configuration
app.config['GOOGLE_CLIENT_ID'] = os.getenv('GOOGLE_CLIENT_ID')
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv('GOOGLE_CLIENT_SECRET')
app.config['GOOGLE_REDIRECT_URI'] = os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:5000/google-callback')

# Ensure the redirect URI is properly set
if not app.config['GOOGLE_CLIENT_ID'] or not app.config['GOOGLE_CLIENT_SECRET']:
    logger.warning("Google OAuth credentials not configured. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env file")

# MongoDB configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB = os.getenv('MONGODB_DB', 'cyberlaw_assistant')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]

# Initialize login manager
login_manager.init_app(app)

# Add UTC timezone constant
UTC = datetime.UTC

# Initialize database collections
def init_db(app):
    """Initialize database collections and indexes"""
    # Create collections if they don't exist
    collections = {
        'users': {
            'indexes': [
                ('username', {'unique': True}),
                ('email', {'unique': True})
            ]
        },
        'conversations': {
            'indexes': [
                ('user_id', {}),
                ('created_at', {'expireAfterSeconds': 30 * 24 * 60 * 60})  # 30 days TTL
            ]
        },
        'messages': {
            'indexes': [
                ('conversation_id', {}),
                ('timestamp', {'expireAfterSeconds': 30 * 24 * 60 * 60})  # 30 days TTL
            ]
        },
        'feedback': {
            'indexes': [
                ('message_id', {}),
                ('created_at', {})
            ]
        },
        'faqs': {
            'indexes': [
                ('category', {}),
                ('tags', {})
            ]
        },
        'legal_documents': {
            'indexes': [
                ('document_type', {}),
                ('section', {}),
                ('tags', {})
            ]
        },
        'training_data': {
            'indexes': [
                ('feedback_id', {}),
                ('created_at', {})
            ]
        },
        'contact_messages': {
            'indexes': [
                ('email', {}),
                ('created_at', {'expireAfterSeconds': 90 * 24 * 60 * 60})  # 90 days TTL
            ]
        }
    }
    
    for collection_name, config in collections.items():
        if collection_name not in db.list_collection_names():
            db.create_collection(collection_name)
        
        # Create indexes
        for field, options in config['indexes']:
            db[collection_name].create_index(field, **options)
    
    # Add admin user if it doesn't exist
    if not db.users.find_one({'username': 'admin'}):
        db.users.insert_one({
            'username': 'admin',
            'email': 'admin@example.com',
            'password': generate_password_hash('admin'),
            'created_at': datetime.datetime.now(UTC),
            'role': 'admin'
        })

# Initialize database
with app.app_context():
    init_db(app)
    # Initialize analytics dashboard
    analytics_app = create_dashboard(app)

# Update User class for MongoDB
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.email = user_data['email']

@login_manager.user_loader
def load_user(user_id):
    user_data = db.users.find_one({'_id': ObjectId(user_id)})
    if user_data:
        return User(user_data)
    return None

def register_user(username, email, password):
    try:
        # Check if user already exists
        if db.users.find_one({'$or': [{'username': username}, {'email': email}]}):
            return False, "Username or email already exists"
        
        # Hash password and insert new user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        db.users.insert_one({
            'username': username,
            'email': email,
            'password': hashed_password,
            'created_at': datetime.datetime.now(UTC)
        })
        return True, "Registration successful"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def authenticate_user(username_or_email, password):
    try:
        # Check if input is email or username
        query = {'email': username_or_email} if '@' in username_or_email else {'username': username_or_email}
        user_data = db.users.find_one(query)
        
        if user_data and check_password_hash(user_data['password'], password):
            return User(user_data)
        return None
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        return None

def save_conversation(user_id, title, messages):
    """Save a conversation with its messages"""
    try:
        # Create conversation
        conversation = {
            'user_id': ObjectId(user_id),
            'title': title,
            'created_at': datetime.datetime.now(UTC)
        }
        result = db.conversations.insert_one(conversation)
        conversation_id = result.inserted_id
        
        # Save messages
        message_docs = []
        for message in messages:
            message_docs.append({
                'conversation_id': conversation_id,
                'sender': message['sender'],
                'content': message['content'],
                'timestamp': datetime.datetime.now(UTC)
            })
        if message_docs:
            db.messages.insert_many(message_docs)
        
        return True, str(conversation_id)
    except Exception as e:
        return False, f"Failed to save conversation: {str(e)}"

def get_user_conversations(user_id):
    """Get all conversations for a user"""
    try:
        conversations = list(db.conversations.find(
            {'user_id': ObjectId(user_id)},
            {'_id': 1, 'title': 1, 'created_at': 1}
        ).sort('created_at', -1))
        
        # Add message count to each conversation
        for conv in conversations:
            conv['message_count'] = db.messages.count_documents({'conversation_id': conv['_id']})
            conv['_id'] = str(conv['_id'])
            conv['created_at'] = conv['created_at'].isoformat()
        
        return conversations
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        return []

def get_conversation_messages(conversation_id, user_id):
    """Get all messages for a conversation"""
    try:
        # First verify that the conversation belongs to the user
        conversation = db.conversations.find_one({
            '_id': ObjectId(conversation_id),
            'user_id': ObjectId(user_id)
        })
        
        if not conversation:
            return None  # User doesn't own this conversation
        
        messages = list(db.messages.find(
            {'conversation_id': ObjectId(conversation_id)}
        ).sort('timestamp', 1))
        
        # Convert ObjectId to string and datetime to ISO format
        for msg in messages:
            msg['_id'] = str(msg['_id'])
            msg['timestamp'] = msg['timestamp'].isoformat()
        
        return messages
    except Exception as e:
        logger.error(f"Error getting messages: {str(e)}")
        return None

def save_feedback(message_id, rating, comment=None):
    """Save feedback for a bot message"""
    try:
        db.feedback.insert_one({
            'message_id': ObjectId(message_id),
            'rating': rating,
            'comment': comment,
            'created_at': datetime.datetime.now(UTC)
        })
        return True
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        return False

# Add intent detection patterns
INTENT_PATTERNS = {
    'greeting': r'\b(hello|hi|hey|greetings|good\s(morning|afternoon|evening)|namaste|namaskar)\b',
    'legislation': r'\b(law|act|legislation|section|statute|regulation)\b',
    'reporting': r'\b(report|complain|file|helpline|contact|notify)\b',
    'enforcement': r'\b(agency|authority|enforcement|police|investigate|prosecute)\b',
    'prevention': r'\b(prevent|protect|secure|safeguard|defense|security)\b',
    'penalties': r'\b(penalty|punishment|fine|jail|imprisonment|consequence)\b',
    'procedure': r'\b(procedure|process|step|method|way|how to)\b'
}

# Add prompt templates for different intents
PROMPT_TEMPLATES = {
    'greeting': """Context: {context}

{history_context}

User Query: {user_input}

Please provide a friendly greeting response. Structure your response as follows:

1. Greeting: [Warm welcome]
2. Introduction: [Brief introduction of the assistant]
3. Purpose: [Explain what you can help with]
4. Offer: [Ask how you can assist with cyber law queries]""",

    'legislation': """Context: {context}

{history_context}

User Query: {user_input}

Please provide a detailed response focusing on Indian cyber law legislation. Structure your response as follows:

1. Relevant Legislation: [List applicable laws]
2. Key Sections: [Important sections and their implications]
3. Legal Framework: [How these laws work together]
4. Recent Updates: [Any recent amendments or changes]
5. Practical Implications: [How this affects users]""",

    'reporting': """Context: {context}

{history_context}

User Query: {user_input}

Please provide a detailed response focusing on cyber crime reporting procedures. Structure your response as follows:

1. Reporting Channels: [Available reporting methods]
2. Required Information: [What to prepare]
3. Step-by-Step Process: [Detailed reporting steps]
4. Timeline: [Expected response time]
5. Follow-up: [What happens after reporting]""",

    'enforcement': """Context: {context}

{history_context}

User Query: {user_input}

Please provide a detailed response focusing on cyber law enforcement. Structure your response as follows:

1. Enforcement Agencies: [Relevant authorities]
2. Jurisdiction: [Who handles what]
3. Investigation Process: [How cases are handled]
4. Legal Powers: [Authority and limitations]
5. Cooperation: [How to work with authorities]""",

    'prevention': """Context: {context}

{history_context}

User Query: {user_input}

Please provide a detailed response focusing on cyber crime prevention. Structure your response as follows:

1. Risk Assessment: [Common threats]
2. Preventive Measures: [Practical steps]
3. Best Practices: [Security guidelines]
4. Tools & Resources: [Available protection]
5. Regular Updates: [Keeping safe]""",

    'penalties': """Context: {context}

{history_context}

User Query: {user_input}

Please provide a detailed response focusing on cyber law penalties. Structure your response as follows:

1. Applicable Penalties: [Types of punishments]
2. Legal Basis: [Relevant sections]
3. Severity Levels: [Different degrees]
4. Mitigating Factors: [What can help]
5. Case Examples: [Real cases]""",

    'procedure': """Context: {context}

{history_context}

User Query: {user_input}

Please provide a detailed response focusing on cyber law procedures. Structure your response as follows:

1. Required Steps: [Process overview]
2. Documentation: [What to prepare]
3. Timeline: [Expected duration]
4. Authorities: [Who to contact]
5. Follow-up: [Next steps]"""
}

# Add default template for general queries
DEFAULT_TEMPLATE = """Context: {context}

{history_context}

User Query: {user_input}

Please provide a detailed response focusing on Indian cyber law aspects. Structure your response as follows:

1. Main Topic: [Brief overview]
2. Key Points: [Important aspects]
3. Legal Framework: [Relevant laws]
4. Practical Guidance: [What to do]
5. Additional Resources: [Where to learn more]"""

def detect_intent(user_input):
    """Detect the intent of the user query using regex patterns"""
    input_lower = user_input.lower()
    detected_intents = []
    
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, input_lower):
            detected_intents.append(intent)
    
    return detected_intents[0] if detected_intents else None

def get_prompt_template(intent):
    """Get the appropriate prompt template based on intent"""
    return PROMPT_TEMPLATES.get(intent, DEFAULT_TEMPLATE)

# Add caching for frequently asked questions
@lru_cache(maxsize=100)
def get_cached_response(query_hash):
    """Get cached response for frequently asked questions"""
    return None  # Implement actual caching logic here

def generate_query_hash(user_input, context):
    """Generate a hash for the query and context"""
    combined = f"{user_input}:{json.dumps(context, sort_keys=True)}"
    return hashlib.md5(combined.encode()).hexdigest()

class ResponseGenerator:
    """Response generator for cyber law queries"""
    
    def __init__(self):
        self.service_url = os.getenv('LLM_SERVICE_URL', 'http://localhost:11434')
        self.model_name = os.getenv('LLM_MODEL_NAME', 'llama3.2:latest')
        self.service_available = False
        self.cache = {}
        
        # Test service connection and model availability
        try:
            logger.info(f"Attempting to connect to service at {self.service_url}")
            response = requests.get(f"{self.service_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json().get("models", [])]
                if self.model_name in available_models:
                    logger.info(f"Successfully connected to service with model {self.model_name}")
                    self.service_available = True
                else:
                    logger.warning(f"Model {self.model_name} not available. Available models: {available_models}")
                    if available_models:
                        self.model_name = available_models[0]
                        logger.info(f"Using fallback model: {self.model_name}")
                        self.service_available = True
                    else:
                        logger.error("No suitable models available")
                        self.service_available = False
            else:
                logger.error(f"Failed to connect to service: {response.status_code}")
                self.service_available = False
        except Exception as e:
            logger.error(f"Unexpected error during service connection: {str(e)}")
            self.service_available = False

    def generate_response(self, user_input, context, conversation_history=None):
        """Generate response with improved error handling and conversation context"""
        if not self.service_available:
            logger.error("Service not available. Please check if the service is running.")
            return {
                "status": "error",
                "message": "Service not available. Please check if the service is running."
            }
            
        try:
            # Check cache first
            query_hash = generate_query_hash(user_input, context)
            cached_response = get_cached_response(query_hash)
            if cached_response:
                logger.info("Returning cached response")
                return cached_response

            # Detect intent and get appropriate template
            intent = detect_intent(user_input)
            template = get_prompt_template(intent)
            
            # Prepare conversation history context if provided
            history_context = ""
            if conversation_history and len(conversation_history) > 0:
                history_context = "Previous conversation:\n"
                for msg in conversation_history[-3:]:  # Use last 3 messages for context
                    role = "User" if msg.get('sender') == 'user' else "Assistant"
                    history_context += f"{role}: {msg.get('content')}\n"
            
            # Prepare the prompt with context and history
            prompt = template.format(
                context=context,
                history_context=history_context,
                user_input=user_input
            )
            
            logger.info(f"Sending request to service with prompt length: {len(prompt)}")
            
            # Make API request with retry logic
            max_retries = 3
            retry_delay = 2  # seconds
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{self.service_url}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": float(os.getenv('LLM_TEMPERATURE', '0.7')),
                                "top_p": float(os.getenv('LLM_TOP_P', '0.9')),
                                "max_tokens": int(os.getenv('LLM_MAX_TOKENS', '1000')),
                                "presence_penalty": 0.6,
                                "frequency_penalty": 0.3
                            }
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        response_text = result.get("response", "").strip()
                        
                        if not response_text:
                            logger.error("Empty response from service")
                            return {
                                "status": "error",
                                "message": "Received empty response. Please try again."
                            }
                        
                        # Parse the response into structured format
                        sections = response_text.split('\n\n')
                        structured_content = []
                        
                        for section in sections:
                            if section.strip():
                                lines = section.strip().split('\n')
                                title = lines[0].strip()
                                content = []
                                
                                for line in lines[1:]:
                                    line = line.strip()
                                    if line.startswith('- '):
                                        content.append({
                                            "type": "list_item",
                                            "content": line[2:].strip()
                                        })
                                    else:
                                        content.append({
                                            "type": "paragraph",
                                            "content": line.strip()
                                        })
                                
                                structured_content.append({
                                    "type": "section",
                                    "title": title,
                                    "content": content
                                })
                        
                        response_data = {
                            "status": "success",
                            "response": {
                                "type": "structured",
                                "content": structured_content
                            },
                            "context": context,
                            "timestamp": datetime.datetime.now(UTC).isoformat()
                        }
                        
                        # Cache the response
                        self.cache[query_hash] = response_data
                        
                        # Store in training data collection for future improvements
                        db.training_data.insert_one({
                            'query': user_input,
                            'response': response_data,
                            'feedback_id': None,  # Will be updated when feedback is received
                            'created_at': datetime.datetime.now(UTC)
                        })
                        
                        return response_data
                    else:
                        logger.error(f"Service error: {response.status_code} - {response.text}")
                        last_error = f"Error generating response (HTTP {response.status_code}). Please try again."
                        
                except requests.exceptions.Timeout:
                    logger.error(f"Service request timed out (attempt {attempt + 1}/{max_retries})")
                    last_error = "Request timed out. Please try again later."
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                except requests.exceptions.ConnectionError:
                    logger.error("Failed to connect to service")
                    last_error = "Failed to connect to the service. Please try again later."
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    last_error = "An unexpected error occurred. Please try again later."
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                        
            # If we get here, all retries failed
            return {
                "status": "error",
                "message": last_error
            }
                
        except Exception as e:
            logger.error(f"Unexpected error in response generation: {e}")
            return {
                "status": "error",
                "message": "An unexpected error occurred. Please try again later."
            }

# Knowledge base for Indian cyber law
INDIAN_CYBER_LAW_KB = {
    "primary_legislation": {
        "IT_Act_2000": {
            "title": "Information Technology Act, 2000",
            "description": "Primary legislation governing cyber activities in India",
            "key_sections": {
                "43": "Penalty for damage to computer, computer system, etc.",
                "66": "Computer related offences",
                "66F": "Cyber terrorism",
                "67": "Publishing or transmitting obscene material in electronic form"
            }
        },
        "DPDP_2023": {
            "title": "Digital Personal Data Protection Act, 2023",
            "description": "Regulates processing of digital personal data in India"
        }
    },
    "reporting_mechanisms": {
        "portal": "National Cyber Crime Reporting Portal (cybercrime.gov.in)",
        "helpline": "Cyber Crime Helpline: 1930",
        "email": "CERT-In: incident@cert-in.org.in"
    },
    "enforcement_agencies": [
        "Cyber Crime Police Stations",
        "CERT-In (Indian Computer Emergency Response Team)",
        "Cyber Appellate Tribunal",
        "National Critical Information Infrastructure Protection Centre (NCIIPC)"
    ]
}

def get_relevant_context(user_input):
    """Extract relevant context from knowledge base based on user input"""
    input_lower = user_input.lower()
    context = {}
    
    # Check for legislation-related queries
    if any(keyword in input_lower for keyword in ["law", "act", "legislation", "section"]):
        context["legislation"] = INDIAN_CYBER_LAW_KB["primary_legislation"]
    
    # Check for reporting-related queries
    if any(keyword in input_lower for keyword in ["report", "complain", "file", "helpline"]):
        context["reporting"] = INDIAN_CYBER_LAW_KB["reporting_mechanisms"]
    
    # Check for agency-related queries
    if any(keyword in input_lower for keyword in ["agency", "authority", "enforcement", "police"]):
        context["agencies"] = INDIAN_CYBER_LAW_KB["enforcement_agencies"]
    
    # If no specific context found, provide general knowledge
    if not context:
        context = INDIAN_CYBER_LAW_KB
    
    return context

# Initialize response generator
response_generator = ResponseGenerator()

# Add custom template filter for JSON
@app.template_filter('from_json')
def from_json(value):
    try:
        return json.loads(value)
    except:
        return value

# Route for handling login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        logger.info(f"User {current_user.username} is already authenticated, redirecting to index")
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        logger.info(f"Login attempt for user: {username}")
        
        if not username or not password:
            logger.warning("Login attempt with missing credentials")
            flash('Please enter both username and password', 'error')
            return render_template('login.html')
        
        user = authenticate_user(username, password)
        if user:
            login_user(user, remember=True)
            next_page = request.args.get('next')
            logger.info(f"Login successful for user: {username}, redirecting to: {next_page or 'index'}")
            
            # Ensure next_page is safe (starts with /)
            if next_page and not next_page.startswith('/'):
                next_page = None
                
            return redirect(next_page or url_for('index'))
        else:
            logger.warning(f"Failed login attempt for user: {username}")
            flash('Invalid username or password', 'error')
    
    # Log the next parameter for debugging
    next_page = request.args.get('next')
    if next_page:
        logger.info(f"Login page accessed with next parameter: {next_page}")
    
    return render_template('login.html')

# Route for handling registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not all([username, email, password, confirm_password]):
            flash('Please fill out all fields', 'error')
            return render_template('register.html')
            
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
            
        success, message = register_user(username, email, password)
        if success:
            flash(message, 'success')
            return redirect(url_for('login'))
        else:
            flash(message, 'error')
    
    return render_template('register.html')

# Route for logging out
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

# Route for user dashboard
@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's recent conversations
    conversations = list(db.conversations.find(
        {'user_id': ObjectId(current_user.id)}
    ).sort('created_at', -1).limit(5))
    
    # Get user data
    user_data = db.users.find_one({'_id': ObjectId(current_user.id)})
    
    # Format dates and add message counts
    for conv in conversations:
        conv['created_at'] = conv['created_at'].strftime('%Y-%m-%d %H:%M')
        conv['message_count'] = db.messages.count_documents({'conversation_id': conv['_id']})
        conv['_id'] = str(conv['_id'])  # Convert ObjectId to string for template rendering
    
    return render_template('dashboard.html', 
                         conversations=conversations,
                         user_data=user_data)

@app.route('/export-data')
@login_required
def export_data():
    try:
        # Get user data
        user_data = db.users.find_one({'_id': ObjectId(current_user.id)})
        if not user_data:
            raise Exception("User data not found")

        # Get all user's conversations
        conversations = list(db.conversations.find(
            {'user_id': ObjectId(current_user.id)}
        ).sort('created_at', -1))
        
        # Get all messages for these conversations
        conversation_data = []
        for conv in conversations:
            messages = list(db.messages.find(
                {'conversation_id': conv['_id']}
            ).sort('timestamp', 1))
            
            # Format conversation data
            conv_data = {
                'title': conv['title'],
                'created_at': conv['created_at'].isoformat(),
                'messages': []
            }
            
            # Format messages
            for msg in messages:
                try:
                    # Handle both string and JSON content
                    content = msg['content']
                    if isinstance(content, str):
                        try:
                            content = json.loads(content)
                        except json.JSONDecodeError:
                            pass
                    
                    conv_data['messages'].append({
                        'sender': msg['sender'],
                        'content': content,
                        'timestamp': msg['timestamp'].isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Error processing message: {str(e)}")
                    continue
            
            conversation_data.append(conv_data)
        
        # Create export data
        export_data = {
            'user': {
                'username': current_user.username,
                'email': current_user.email,
                'created_at': user_data['created_at'].isoformat() if user_data and user_data.get('created_at') else None
            },
            'conversations': conversation_data,
            'export_date': datetime.datetime.now(UTC).isoformat()
        }
        
        # Set response headers for file download
        response = jsonify(export_data)
        response.headers['Content-Disposition'] = f'attachment; filename=cyberlaw_assistant_data_{datetime.datetime.now(UTC).strftime("%Y%m%d_%H%M%S")}.json'
        response.headers['Content-Type'] = 'application/json'
        
        return response
        
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to export data: {str(e)}'
        }), 500

@app.route('/clear-history', methods=['POST'])
@login_required
def clear_history():
    try:
        # Get user ID and convert to ObjectId
        user_id = ObjectId(current_user.id)
        logger.info(f"Attempting to clear history for user: {current_user.username} (ID: {user_id})")
        
        # First get all conversation IDs for this user
        conversations = list(db.conversations.find(
            {'user_id': user_id},
            {'_id': 1}
        ))
        conversation_ids = [conv['_id'] for conv in conversations]
        
        logger.info(f"Found {len(conversation_ids)} conversations to delete")
        
        # Delete all messages for these conversations
        if conversation_ids:
            messages_result = db.messages.delete_many({
                'conversation_id': {'$in': conversation_ids}
            })
            logger.info(f"Deleted {messages_result.deleted_count} messages")
        
        # Delete all conversations for this user
        conversations_result = db.conversations.delete_many({'user_id': user_id})
        logger.info(f"Deleted {conversations_result.deleted_count} conversations")
        
        # Verify deletion
        remaining_conversations = db.conversations.count_documents({'user_id': user_id})
        remaining_messages = db.messages.count_documents({
            'conversation_id': {'$in': conversation_ids}
        })
        
        if remaining_conversations > 0 or remaining_messages > 0:
            logger.error(f"Deletion verification failed: {remaining_conversations} conversations and {remaining_messages} messages still exist")
            return jsonify({
                'status': 'error',
                'message': 'Some conversations or messages could not be deleted'
            }), 500
        
        logger.info("Successfully cleared all history")
        return jsonify({'status': 'success'})
        
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to clear history: {str(e)}'
        }), 500

# Route for viewing a conversation
@app.route('/conversation/<string:conversation_id>')
@login_required
def view_conversation(conversation_id):
    messages = get_conversation_messages(conversation_id, current_user.id)
    if messages is None:
        flash('Conversation not found or access denied', 'error')
        return redirect(url_for('dashboard'))
    
    return render_template('conversation.html', messages=messages)

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    """Handle chat requests with user authentication and chat history"""
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        conversation_id = data.get('conversation_id')
        
        if not user_input:
            return jsonify({
                "status": "error",
                "message": "Please provide a valid message."
            })
        
        # Get conversation history if available
        conversation_history = []
        if conversation_id:
            conversation_history = get_conversation_messages(conversation_id, current_user.id)
        
        # Get relevant context from knowledge base
        context = get_relevant_context(user_input)
        
        # Generate response using the response generator
        response = response_generator.generate_response(user_input, context, conversation_history)
        
        # Save the conversation
        if not conversation_id:
            # Generate title from user input (first few words)
            title = user_input[:30] + "..." if len(user_input) > 30 else user_input
            
            # Save conversation and messages
            messages = [
                {"sender": "user", "content": user_input},
                {"sender": "bot", "content": json.dumps(response.get("response", {}))}
            ]
            success, new_conversation_id = save_conversation(current_user.id, title, messages)
            
            if success:
                response["conversation_id"] = new_conversation_id
        else:
            # Add message to existing conversation
            # This would require adding a function to add single messages to a conversation
            pass
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "An error occurred while processing your request."
        })

# Route for submitting feedback
@app.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    try:
        data = request.get_json()
        message_id = data.get('message_id')
        rating = data.get('rating')
        comment = data.get('comment', '')
        
        if not message_id or not rating:
            return jsonify({
                "status": "error",
                "message": "Message ID and rating are required."
            })
        
        # Validate rating
        valid_ratings = ['helpful', 'not_helpful', 'report']
        if rating not in valid_ratings:
            return jsonify({
                "status": "error",
                "message": "Invalid rating value."
            })
        
        # Convert rating to numeric value for storage
        rating_value = 1 if rating == 'helpful' else -1 if rating == 'not_helpful' else 0
        
        success = save_feedback(message_id, rating_value, comment)
        if success:
            return jsonify({
                "status": "success",
                "message": "Feedback submitted successfully."
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to save feedback."
            })
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "An error occurred while saving your feedback."
        })

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        if not all([name, email, subject, message]):
            flash('Please fill out all fields', 'error')
            return render_template('contact.html')
        
        try:
            # Store the contact message in MongoDB
            contact_message = {
                'name': name,
                'email': email,
                'subject': subject,
                'message': message,
                'created_at': datetime.datetime.now(UTC),
                'status': 'new',  # new, read, responded
                'ip_address': request.remote_addr
            }
            
            result = db.contact_messages.insert_one(contact_message)
            
            if result.inserted_id:
                logger.info(f"Contact form submission stored successfully from {name} ({email}): {subject}")
                flash('Thank you for your message. We will get back to you soon!', 'success')
                return redirect(url_for('contact'))
            else:
                raise Exception("Failed to store contact message")
                
        except Exception as e:
            logger.error(f"Error processing contact form: {str(e)}")
            flash('An error occurred while sending your message. Please try again.', 'error')
    
    return render_template('contact.html')

@app.route('/debug/conversations')
@login_required
def debug_conversations():
    try:
        # Get all conversations for the current user
        conversations = list(db.conversations.find(
            {'user_id': ObjectId(current_user.id)}
        ).sort('created_at', -1))
        
        # Get messages for each conversation
        for conv in conversations:
            conv['_id'] = str(conv['_id'])
            conv['user_id'] = str(conv['user_id'])
            conv['created_at'] = conv['created_at'].isoformat()
            
            # Get messages for this conversation
            messages = list(db.messages.find(
                {'conversation_id': ObjectId(conv['_id'])}
            ).sort('timestamp', 1))
            
            # Convert message IDs and timestamps
            for msg in messages:
                msg['_id'] = str(msg['_id'])
                msg['conversation_id'] = str(msg['conversation_id'])
                msg['timestamp'] = msg['timestamp'].isoformat()
            
            conv['messages'] = messages
        
        return jsonify({
            'status': 'success',
            'conversations': conversations
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/google-login')
def google_login():
    """Initiate Google OAuth login"""
    try:
        if not app.config['GOOGLE_CLIENT_ID'] or not app.config['GOOGLE_CLIENT_SECRET']:
            logger.error("Google OAuth credentials not configured")
            flash('Google login is not configured. Please contact the administrator.', 'error')
            return redirect(url_for('login'))

        # Create OAuth2 flow
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": app.config['GOOGLE_CLIENT_ID'],
                    "project_id": "cyberlaw-assistant",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_secret": app.config['GOOGLE_CLIENT_SECRET'],
                    "redirect_uris": [app.config['GOOGLE_REDIRECT_URI']],
                    "javascript_origins": ["http://localhost:5000"]
                }
            },
            scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile']
        )
        
        # Set the redirect URI explicitly
        flow.redirect_uri = app.config['GOOGLE_REDIRECT_URI']
        
        # Generate authorization URL
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        
        # Store state in session
        session['state'] = state
        session['flow'] = flow
        
        logger.info(f"Redirecting to Google authorization URL: {authorization_url}")
        return redirect(authorization_url)
        
    except Exception as e:
        logger.error(f"Google login error: {str(e)}")
        flash('Error initiating Google login. Please try again.', 'error')
        return redirect(url_for('login'))

@app.route('/google-callback')
def google_callback():
    """Handle Google OAuth callback"""
    try:
        if 'state' not in session:
            logger.error("No state found in session")
            flash('Invalid session state. Please try logging in again.', 'error')
            return redirect(url_for('login'))

        flow = session.get('flow')
        if not flow:
            logger.error("No flow found in session")
            flash('Invalid session. Please try logging in again.', 'error')
            return redirect(url_for('login'))

        # Exchange authorization code for credentials
        flow.fetch_token(
            authorization_response=request.url,
            include_granted_scopes='true'
        )
        
        credentials = flow.credentials
        request_session = requests.session()
        token_request = google_requests.Request(session=request_session)
        
        # Verify the token
        id_info = google.oauth2.id_token.verify_oauth2_token(
            credentials.id_token, token_request, app.config['GOOGLE_CLIENT_ID']
        )
        
        # Check if user exists
        user_data = db.users.find_one({'email': id_info['email']})
        
        if not user_data:
            # Create new user
            user_data = {
                'username': id_info['email'].split('@')[0],
                'email': id_info['email'],
                'password': generate_password_hash(os.urandom(24).hex()),  # Random password for OAuth users
                'created_at': datetime.datetime.now(UTC),
                'google_id': id_info['sub'],
                'name': id_info.get('name', ''),
                'picture': id_info.get('picture', '')
            }
            result = db.users.insert_one(user_data)
            user_data['_id'] = result.inserted_id
            logger.info(f"Created new user from Google login: {user_data['email']}")
        
        # Log in the user
        user = User(user_data)
        login_user(user)
        
        # Clear session data
        session.pop('state', None)
        session.pop('flow', None)
        
        logger.info(f"User logged in successfully via Google: {user.email}")
        flash('Successfully logged in with Google!', 'success')
        return redirect(url_for('index'))
        
    except Exception as e:
        logger.error(f"Google callback error: {str(e)}")
        flash('Error during Google authentication. Please try again.', 'error')
        return redirect(url_for('login'))

@app.route('/analytics')
@login_required
def analytics():
    """Redirect to the analytics dashboard"""
    return redirect('/analytics/')

if __name__ == '__main__':
    app.run(debug=True)