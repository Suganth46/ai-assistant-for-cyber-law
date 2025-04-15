# Upgraded Cyber Law Assistant
# Enhanced with modern NLP techniques including transformers, semantic search, and NER

import json
import os
import random
import datetime
import re
from collections import defaultdict
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

# Import NLP libraries
try:
    import torch
    from transformers import AutoTokenizer, AutoModel, pipeline
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc
except ImportError:
    print("Please install required libraries with: pip install torch transformers sentence-transformers scikit-learn spacy")
    print("Also download spaCy model with: python -m spacy download en_core_web_sm")

# Set up logging
logging.basicConfig(
    filename='cyber_law_assistant.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NLPEngine:
    """Advanced NLP processing engine using transformers and other techniques"""
    
    def __init__(self, use_gpu: bool = False):
        """Initialize NLP components"""
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load sentence embedding model
        try:
            self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            self.embedding_model.to(self.device)
            logger.info("Sentence embedding model loaded")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.embedding_model = None
            
        # Load spaCy for NER and linguistic processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
            # Add custom pipeline components
            self.add_custom_components()
            logger.info("SpaCy NLP model loaded")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            self.nlp = None
            
        # Initialize intent classifier
        try:
            self.intent_classifier = pipeline("zero-shot-classification", 
                                             model="facebook/bart-large-mnli", 
                                             device=0 if self.device.type == "cuda" else -1)
            logger.info("Intent classifier loaded")
        except Exception as e:
            logger.error(f"Error loading intent classifier: {e}")
            self.intent_classifier = None
            
        # Initialize summarizer
        try:
            self.summarizer = pipeline("summarization", 
                                      model="facebook/bart-large-cnn", 
                                      device=0 if self.device.type == "cuda" else -1)
            logger.info("Text summarizer loaded")
        except Exception as e:
            logger.error(f"Error loading summarizer: {e}")
            self.summarizer = None
    
    def add_custom_components(self):
        """Add custom pipeline components to spaCy"""
        if not self.nlp:
            return
            
        # Add cyber law entity recognition component
        @Language.component("cyber_law_entities")
        def cyber_law_entities(doc: Doc) -> Doc:
            """Add custom entity recognition for cyber law terms"""
            # Define cyber law specific terms and their labels
            cyber_law_terms = {
                "hacking": "CYBER_CRIME",
                "phishing": "CYBER_CRIME",
                "data breach": "CYBER_CRIME",
                "malware": "CYBER_CRIME",
                "ransomware": "CYBER_CRIME",
                "identity theft": "CYBER_CRIME",
                "IT Act": "LEGISLATION",
                "Information Technology Act": "LEGISLATION",
                "POCSO": "LEGISLATION",
                "NCIIPC": "ORGANIZATION",
                "cyber crime": "CYBER_CRIME",
                "cybercrime": "CYBER_CRIME",
                "cyber law": "LEGAL_TERM",
                "cyber security": "CONCEPT"
            }
            
            # Find matches in text and add entities
            text = doc.text.lower()
            for term, label in cyber_law_terms.items():
                for match in re.finditer(r'\b' + re.escape(term) + r'\b', text):
                    start, end = match.span()
                    # Find token indices
                    start_char = match.start()
                    end_char = match.end()
                    start_token = None
                    end_token = None
                    
                    for i, token in enumerate(doc):
                        if token.idx <= start_char < token.idx + len(token.text):
                            start_token = i
                        if token.idx <= end_char <= token.idx + len(token.text) and end_token is None:
                            end_token = i + 1
                            
                    if start_token is not None and end_token is not None:
                        ent = doc.char_span(start_char, end_char, label=label)
                        if ent is not None:
                            doc.ents = list(doc.ents) + [ent]
                            
            return doc
            
        # Add component to pipeline
        self.nlp.add_pipe("cyber_law_entities", after="ner")
    
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)"""
        if not self.embedding_model:
            logger.warning("Embedding model not available")
            # Return zero vectors as fallback
            if isinstance(texts, str):
                return np.zeros(384)  # Default dimension for MiniLM model
            else:
                return np.zeros((len(texts), 384))
        
        try:
            return self.embedding_model.encode(texts, convert_to_numpy=True, 
                                              device=self.device, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero vectors as fallback
            if isinstance(texts, str):
                return np.zeros(384)  # Default dimension for MiniLM model
            else:
                return np.zeros((len(texts), 384))
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate the semantic similarity between two texts"""
        try:
            embedding1 = self.get_embeddings(text1).reshape(1, -1)
            embedding2 = self.get_embeddings(text2).reshape(1, -1)
            
            return cosine_similarity(embedding1, embedding2)[0][0]
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def classify_intent(self, text: str, candidate_intents: List[str]) -> Tuple[str, float]:
        """Classify the intent of a text using zero-shot classification"""
        if not self.intent_classifier or not candidate_intents:
            # Fallback to basic keyword matching
            return self._fallback_intent_classification(text, candidate_intents)
            
        try:
            result = self.intent_classifier(text, candidate_intents)
            top_intent = result["labels"][0]
            confidence = result["scores"][0]
            return top_intent, confidence
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return self._fallback_intent_classification(text, candidate_intents)
    
    def _fallback_intent_classification(self, text: str, candidate_intents: List[str]) -> Tuple[str, float]:
        """Simple fallback when transformer model is unavailable"""
        text = text.lower()
        best_intent = candidate_intents[0]
        best_score = 0
        
        for intent in candidate_intents:
            # Count word overlap
            intent_words = intent.lower().split("_")
            score = sum(1 for word in intent_words if word in text) / len(intent_words)
            
            if score > best_score:
                best_score = score
                best_intent = intent
                
        return best_intent, best_score
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text using spaCy"""
        if not self.nlp:
            return {}
            
        try:
            doc = self.nlp(text)
            
            # Group entities by label
            entities = defaultdict(list)
            for ent in doc.ents:
                entities[ent.label_].append(ent.text)
                
            return dict(entities)
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {}
    
    def summarize_text(self, text: str, max_length: int = 100, min_length: int = 30) -> str:
        """Generate a concise summary of longer text"""
        if not self.summarizer:
            # Fallback to simple extraction
            return self._fallback_summarization(text, max_length)
            
        # Only attempt to summarize if text is long enough
        if len(text.split()) < 50:
            return text
            
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, 
                                     do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return self._fallback_summarization(text, max_length)
    
    def _fallback_summarization(self, text: str, max_length: int) -> str:
        """Simple fallback summarization"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) <= 3:
            return text
            
        return " ".join(sentences[:3]) + "..."
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze the sentiment and emotional content of text"""
        if not self.nlp:
            return {"sentiment": "neutral", "score": 0.5}
            
        try:
            doc = self.nlp(text)
            
            # Simple rule-based sentiment analysis as fallback
            pos_words = ["good", "great", "excellent", "helpful", "resolved", "solved", 
                        "protect", "secure", "safe", "appreciate", "thanks", "thank"]
            neg_words = ["bad", "terrible", "useless", "unhelpful", "confusing", "difficult", 
                        "problem", "issue", "threat", "attack", "breach", "victim"]
            
            text_lower = text.lower()
            pos_count = sum(1 for word in pos_words if word in text_lower)
            neg_count = sum(1 for word in neg_words if word in text_lower)
            
            total = pos_count + neg_count
            if total == 0:
                sentiment = "neutral"
                score = 0.5
            else:
                score = pos_count / (pos_count + neg_count)
                if score > 0.6:
                    sentiment = "positive"
                elif score < 0.4:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                    
            return {
                "sentiment": sentiment,
                "score": score,
                "pos_count": pos_count,
                "neg_count": neg_count
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"sentiment": "neutral", "score": 0.5}
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        if not text or not isinstance(text, str):
            return ""
            
        # Basic cleanup
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Expand common abbreviations relevant to legal text
        abbreviations = {
            r'\bu/?s\b': 'under section',
            r'\bsec\b': 'section',
            r'\bIT Act\b': 'Information Technology Act',
            r'\bIPC\b': 'Indian Penal Code',
            r'\bNIIA\b': 'National Information Infrastructure Act'
        }
        
        for abbr, expanded in abbreviations.items():
            text = re.sub(abbr, expanded, text, flags=re.IGNORECASE)
            
        return text


class ConversationManager:
    """Enhanced conversation manager with semantic understanding"""

    def __init__(self, max_history=15):
        self.conversation_history = []
        self.max_history = max_history
        self.context = {
            "user_type": None,
            "current_topic": None,
            "last_law_category": None,
            "user_sentiment": "neutral",
            "conversation_phase": "greeting",
            "important_entities": [],
            "session_id": self._generate_session_id()
        }
        self.embedding_cache = {}  # Cache for conversation embeddings
    
    def _generate_session_id(self):
        """Generate a unique session ID"""
        return f"session_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
    
    def add_interaction(self, user_input, assistant_response, nlp_engine=None):
        """Add an interaction to the conversation history with semantic analysis"""
        timestamp = datetime.datetime.now().isoformat()
        
        # Additional NLP analysis if engine is available
        nlp_data = {}
        if nlp_engine:
            # Extract entities
            entities = nlp_engine.extract_entities(user_input)
            
            # Analyze sentiment
            sentiment_data = nlp_engine.analyze_sentiment(user_input)
            
            # Cache embedding for faster similarity search later
            embedding = nlp_engine.get_embeddings(user_input)
            
            nlp_data = {
                "entities": entities,
                "sentiment": sentiment_data,
                "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else None
            }
            
            # Update user sentiment in context
            self.context["user_sentiment"] = sentiment_data["sentiment"]
            
            # Update important entities in context
            for entity_type in ["CYBER_CRIME", "LEGISLATION", "ORG"]:
                if entity_type in entities:
                    for entity in entities[entity_type]:
                        if entity not in self.context["important_entities"]:
                            self.context["important_entities"].append(entity)
        
        interaction = {
            "timestamp": timestamp,
            "user_input": user_input,
            "assistant_response": assistant_response,
            "context": self.context.copy(),
            "nlp_data": nlp_data
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
            
        if self.context["important_entities"]:
            summary.append(f"Important entities mentioned: {', '.join(self.context['important_entities'])}.")
            
        if self.context["user_sentiment"] != "neutral":
            summary.append(f"User sentiment is {self.context['user_sentiment']}.")
        
        return " ".join(summary)
    
    def find_similar_interactions(self, user_input, nlp_engine, top_n=3):
        """Find semantically similar past interactions"""
        if not nlp_engine or len(self.conversation_history) == 0:
            return []
            
        try:
            # Generate embedding for the current input
            current_embedding = nlp_engine.get_embeddings(user_input)
            
            similarities = []
            for i, interaction in enumerate(self.conversation_history):
                # Skip interactions with no embedding data
                if not interaction.get("nlp_data") or not interaction["nlp_data"].get("embedding"):
                    continue
                    
                # Get cached embedding
                past_embedding = np.array(interaction["nlp_data"]["embedding"])
                
                # Calculate similarity
                similarity = cosine_similarity(
                    current_embedding.reshape(1, -1), 
                    past_embedding.reshape(1, -1)
                )[0][0]
                
                similarities.append((i, similarity))
                
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return the top N most similar interactions
            result = []
            for i, sim in similarities[:top_n]:
                interaction = self.conversation_history[i]
                result.append({
                    "user_input": interaction["user_input"],
                    "assistant_response": interaction["assistant_response"],
                    "similarity": sim
                })
                
            return result
                
        except Exception as e:
            logger.error(f"Error finding similar interactions: {e}")
            return []
            
    def determine_conversation_phase(self):
        """Determine the current phase of the conversation"""
        # Simple rule-based approach
        length = len(self.conversation_history)
        
        if length == 0:
            return "greeting"
        elif length == 1:
            return "initial_query"
        elif any(entity in self.context["important_entities"] for entity in ["CYBER_CRIME", "LEGISLATION"]):
            return "specific_guidance"
        elif length > 10:
            return "advanced_discussion"
        else:
            return "exploration"
    
    def save_conversation(self, filepath):
        """Save the conversation history to a file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create a version without the large embedding arrays for storage
            storage_version = []
            for interaction in self.conversation_history:
                # Create a deep copy without the embeddings
                storage_interaction = interaction.copy()
                if "nlp_data" in storage_interaction and "embedding" in storage_interaction["nlp_data"]:
                    # Either remove embedding or store a reduced version
                    storage_interaction["nlp_data"] = storage_interaction["nlp_data"].copy()
                    storage_interaction["nlp_data"]["embedding"] = None
                
                storage_version.append(storage_interaction)
            
            with open(filepath, 'w') as file:
                json.dump({
                    "session_id": self.context["session_id"],
                    "timestamp": datetime.datetime.now().isoformat(),
                    "conversation": storage_version
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
                
                # Rebuild important context from history
                if self.conversation_history:
                    last_interaction = self.conversation_history[-1]
                    if "context" in last_interaction:
                        for key, value in last_interaction["context"].items():
                            self.context[key] = value
            return True
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            return False


class SemanticKnowledgeBase:
    """Enhanced knowledge base with vector search capabilities"""
    
    def __init__(self, knowledge_base_path, nlp_engine):
        self.knowledge_base_path = knowledge_base_path
        self.nlp_engine = nlp_engine
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        self.vector_index = self.build_vector_index()
        
    def load_knowledge_base(self, filepath):
        """Load knowledge base from JSON file"""
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                logger.warning(f"Knowledge base file not found: {filepath}")
                # Try to create directory if needed
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                return self._create_enhanced_knowledge_base()
            
            # Load the file
            with open(filepath, 'r') as file:
                data = json.load(file)
                logger.info(f"Knowledge base loaded successfully from {filepath}")
                return data
                
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return self._create_enhanced_knowledge_base()
    
    def _create_enhanced_knowledge_base(self):
        """Create an enhanced knowledge base structure with comprehensive Indian IT laws"""
        logger.info("Creating enhanced knowledge base structure with comprehensive Indian IT laws")
    
    # Base structure with existing detailed cyber laws
        knowledge_base = {
             "laws": {
                "hacking": {
                "title": "Computer Hacking",
                "description": "Unauthorized access to computer systems, networks, or data with malicious intent or without permission from the owner.",
                "keywords": [
                    "hack", "unauthorized access", "breach", "intrusion", "system compromise", "password cracking", 
                    "cyber attack", "illegal access", "network penetration", "security breach", "data theft",
                    "system infiltration", "computer trespass", "digital break-in", "unauthorized entry",
                    "system cracking", "security bypass", "credential theft", "network infiltration",
                    "cyber intrusion", "malicious access", "computer crime", "system exploitation"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 66 of IT Act, 2000", "Computer Fraud and Abuse Act (CFAA)","Section 43(a) of IT Act, 2000", 
                            "Section 66 of IT Act, 2000",
                            "Section 66B of IT Act, 2000",
                            "Section 66F of IT Act, 2000 (if critical infrastructure)"],
                    "punishment": "Up to 3 years imprisonment or fine up to 5 lakh rupees or both",
                    "aggravating_factors": ["Data theft", "Financial loss", "Critical infrastructure"]
                },
                "reporting_procedure": {
                    "individual": [
                        "Document all evidence of the breach", 
                        "File FIR at local police station or cyber crime portal",
                        "Report to CERT-In if significant breach",
                        "Notify affected parties if personal data was compromised"
                    ],
                    "law_enforcement": [
                        "Gather digital evidence with proper chain of custody",
                        "Consult cyber forensics experts",
                        "Follow jurisdictional procedures",
                        "Consider multi-agency coordination for cross-border cases"
                    ]
                },
                "evidence_collection": [
                    "Server logs showing unauthorized access", 
                    "Access records and timestamps", 
                    "System files with modified timestamps",
                    "Malware or unauthorized software",
                    "Network traffic logs showing suspicious activity"
                ],
                "prevention_tips": [
                    "Implement strong access controls and authentication",
                    "Keep systems and software updated with security patches",
                    "Use firewalls and intrusion detection systems",
                    "Conduct regular security audits and penetration testing",
                    "Train staff on security awareness and social engineering threats"
                ],
                "landmark_cases": [
                    {
                        "name": "State vs John Doe (2020)",
                        "significance": "Established precedent for prosecuting remote hacking across jurisdictions",
                        "key_findings": "Intent to cause damage was proven through digital forensic evidence"
                    },
                    {
                        "name": "United States v. Morris (1991)",
                        "significance": "First major prosecution under the CFAA involving a worm that caused widespread damage",
                        "key_findings": "Unintended consequences of malicious code still constitute criminal liability"
                    }
                ]
            },
            "phishing": {
                "title": "Phishing",
                "description": "Fraudulent attempt to obtain sensitive information by disguising as a trustworthy entity in electronic communications",
                "keywords": [
                    "phish", "fake email", "credential theft", "impersonation", "spoofing", "fraudulent website",
                    "email scam", "identity theft", "fake login", "deceptive communication", "brand impersonation",
                    "social engineering", "account takeover", "fraudulent link", "email forgery", "spoofed domain",
                    "password harvesting", "fake notification", "malicious attachment", "credential phishing",
                    "spear phishing", "whaling", "pharming", "clone phishing", "voice phishing", "smishing",
                    "business email compromise", "CEO fraud", "fake security alert"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 66D of IT Act, 2000", "Identity Theft statutes"],
                    "punishment": "Up to 3 years imprisonment and fine up to 1 lakh rupees",
                    "aggravating_factors": ["Financial fraud", "Identity theft", "Targeting vulnerable populations"]
                },
                "reporting_procedure": {
                    "individual": [
                        "Preserve original phishing email or message with headers",
                        "Report to cyber crime portal or local police",
                        "Contact your bank if financial information was compromised",
                        "Report to email service provider or platform where phishing occurred"
                    ],
                    "law_enforcement": [
                        "Track origin of phishing attempt through email headers",
                        "Coordinate with hosting providers to take down fraudulent websites",
                        "Work with financial institutions if financial fraud involved",
                        "Conduct forensic analysis of phishing infrastructure"
                    ]
                },
                "evidence_collection": [
                    "Original phishing email with complete headers", 
                    "Screenshots of fake websites",
                    "URL links from phishing messages",
                    "Any information submitted to fraudulent sites",
                    "Bank statements showing unauthorized transactions if applicable"
                ],
                "prevention_tips": [
                    "Verify sender email addresses carefully",
                    "Never click on suspicious links in emails or messages",
                    "Check for HTTPS and verify website authenticity before entering credentials",
                    "Use multi-factor authentication wherever possible",
                    "Keep software updated, especially browsers and email clients"
                ],
                "landmark_cases": [
                    {
                        "name": "Cyber Cell v. Anonymous Group (2019)",
                        "significance": "Major phishing operation targeting government officials",
                        "key_findings": "Established liability for sophisticated spear-phishing campaigns"
                    },
                    {
                        "name": "FTC v. Wyndham Worldwide Corp (2015)",
                        "significance": "Established corporate liability for inadequate phishing protection",
                        "key_findings": "Organizations have a duty to protect customers from foreseeable phishing risks"
                    }
                ]
            },
            "data_breach": {
                "title": "Data Breach",
                "description": "Unauthorized access resulting in exposure of sensitive, protected, or confidential data",
                "keywords": [
                    "data leak", "information exposure", "data compromise", "data theft", "unauthorized disclosure",
                    "data spill", "information breach", "confidentiality breach", "data exfiltration", "data loss",
                    "sensitive information theft", "database breach", "unauthorized data access", "data security incident",
                    "records breach", "information leak", "PII exposure", "data breach notification", "compromised records",
                    "data protection failure", "confidential information exposure", "cloud breach", "personal data exposure",
                    "data privacy violation", "customer data leak", "medical records breach", "financial data exposure"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 43A of IT Act, 2000", "Data Protection regulations"],
                    "punishment": "Compensation to affected parties and penalties up to 5 crore rupees depending on severity",
                    "aggravating_factors": ["Negligence", "Delayed notification", "Sensitive personal data"]
                },
                "reporting_procedure": {
                    "individual": [
                        "File complaint with Data Protection Authority",
                        "Notify cyber crime authorities",
                        "Consider civil legal action for damages"
                    ],
                    "law_enforcement": [
                        "Document extent and nature of breached data",
                        "Determine notification requirements based on data type",
                        "Investigate security controls and compliance failures"
                    ]
                },
                "evidence_collection": [
                    "System logs showing unauthorized access",
                    "Documentation of affected records",
                    "Security assessment reports before and after breach",
                    "Communications regarding the breach response",
                    "Records of notification to affected individuals"
                ],
                "prevention_tips": [
                    "Implement encryption for sensitive data",
                    "Conduct regular security assessments",
                    "Limit data collection to necessary information only",
                    "Implement strict access controls and authentication",
                    "Develop and test incident response procedures"
                ]
            },
            "ransomware": {
                "title": "Ransomware Attacks",
                "description": "Malicious software that encrypts victim's data and demands payment for decryption key",
                "keywords": [
                    "ransom", "encryption", "malware", "extortion", "bitcoin", "cryptocurrency",
                    "file encryption", "digital extortion", "crypto malware", "ransom demand", "data hostage",
                    "decryption key", "ransomware infection", "crypto virus", "file hijacking", "system lockout",
                    "data ransom", "malicious encryption", "crypto locker", "payment demand", "data recovery ransom",
                    "double extortion", "threatening notice", "data encryption attack", "locked files", "encrypted data",
                    "ransomware family", "ransomware variant", "ransomware strain", "disk encryption", "system hijack"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 66 and 66F of IT Act, 2000", "Extortion laws"],
                    "punishment": "Up to 3 years imprisonment or fine or both, up to life imprisonment for critical infrastructure",
                    "aggravating_factors": ["Critical infrastructure targeting", "Healthcare targeting", "Public services disruption"]
                },
                "reporting_procedure": {
                    "individual": [
                        "Report to cyber crime authorities immediately",
                        "Contact IT security specialists",
                        "Do not pay ransom without consulting authorities",
                        "Isolate affected systems to prevent spread"
                    ],
                    "law_enforcement": [
                        "Preserve ransom notes and communications",
                        "Track cryptocurrency transactions if payment occurred",
                        "Coordinate with international agencies if needed",
                        "Evaluate national security implications"
                    ]
                },
                "evidence_collection": [
                    "Ransom notes and communications",
                    "Bitcoin wallet addresses used by attackers",
                    "Encrypted file samples",
                    "System logs before encryption",
                    "Malware samples if available"
                ],
                "prevention_tips": [
                    "Maintain regular, offline backups of all critical data",
                    "Keep systems and software updated with security patches",
                    "Implement email filtering to prevent phishing-delivered ransomware",
                    "Use application whitelisting to prevent unauthorized program execution",
                    "Train staff on identifying suspicious emails and attachments"
                ]
            },
            # Adding comprehensive Indian IT laws from the JSON file with enhanced keywords
            "it_act_section_65": {
                "title": "Tampering with Computer Source Code",
                "section": "Section 65 of IT Act, 2000",
                "description": "Knowingly or intentionally concealing, destroying, altering, or causing another to conceal, destroy or alter computer source code required to be kept or maintained by law",
                "keywords": [
                    "source code tampering", "code alteration", "concealing source", "computer program",
                    "source code modification", "program code alteration", "unauthorized code changes",
                    "source code destruction", "code integrity violation", "software source modification",
                    "code concealment", "application source tampering", "program source destruction",
                    "source documentation tampering", "source code manipulation", "algorithm tampering",
                    "code obfuscation", "illegal code modification", "software tampering", "IT Act 65",
                    "source code deletion", "program alteration", "code alteration offense", "source code offense"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 65 of IT Act, 2000"],
                    "punishment": "Imprisonment up to 3 years, or with fine up to ₹2 lakh, or both",
                    "aggravating_factors": ["Commercial intent", "Intellectual property theft", "Causing system failure"]
                }
            },
            "it_act_section_66": {
                "title": "Computer Related Offences",
                "section": "Section 66 of IT Act, 2000",
                "description": "Dishonestly or fraudulently accessing a computer, computer system or network, extracting data, or introducing computer contaminant",
                "keywords": [
                    "unauthorized access", "data theft", "computer fraud", "hacking", "system breach",
                    "fraudulent computer use", "dishonest access", "illegal system access", "computer misuse",
                    "system intrusion", "fraudulent data extraction", "unlawful computer access",
                    "network breach", "unauthorized system use", "computer contaminant", "malicious code",
                    "data extraction", "cyber trespass", "illegal computer operation", "IT Act 66",
                    "fraudulent network access", "unauthorized data access", "computer offense",
                    "cyber offense", "computer system breach", "network intrusion"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 66 of IT Act, 2000"],
                    "punishment": "Imprisonment for a term which may extend to 3 years or with fine which may extend to ₹5 lakh or both",
                    "aggravating_factors": ["Commercial gain", "Critical data compromise", "Repeated offenses"]
                }
            },
            "it_act_section_66a": {
                "title": "Sending offensive messages through communication service",
                "section": "Section 66A of IT Act, 2000",
                "description": "Sending offensive or menacing information through communication services",
                "keywords": [
                    "offensive communication", "electronic message", "threat", "annoyance", "false information",
                    "menacing message", "offensive content", "insulting messages", "grossly offensive",
                    "threatening message", "online harassment", "digital communication offense", "cyber intimidation",
                    "annoying messages", "false communications", "menacing content", "electronic threat",
                    "IT Act 66A", "Shreya Singhal case", "unconstitutional section", "struck down section",
                    "online speech", "communication offense", "digital threat", "invalid section"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 66A of IT Act, 2000"],
                    "punishment": "Note: This section was struck down by the Supreme Court in Shreya Singhal v. Union of India (2015) as unconstitutional",
                    "current_status": "Invalid and unenforceable"
                }
            },
            "it_act_section_66b": {
                "title": "Dishonestly receiving stolen computer resource or communication device",
                "section": "Section 66B of IT Act, 2000",
                "description": "Dishonestly receiving or retaining stolen computer resource or communication device",
                "keywords": [
                    "stolen hardware", "receiving stolen property", "dishonest possession", "stolen devices",
                    "stolen computer", "stolen laptop", "stolen phone", "receiving stolen computer",
                    "dishonest retention", "stolen digital device", "IT equipment theft", "stolen computer parts",
                    "computer theft proceeds", "stolen IT resources", "stolen server", "stolen network device",
                    "IT Act 66B", "receiving stolen computer resource", "handling stolen computer",
                    "stolen device possession", "stolen communications device", "dishonest acquisition",
                    "stolen technology", "retaining stolen computers", "stolen storage device"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 66B of IT Act, 2000"],
                    "punishment": "Imprisonment up to 3 years or fine up to ₹1 lakh or both",
                    "aggravating_factors": ["Commercial intent", "Large volume of devices", "Organized crime connection"]
                }
            },
            "it_act_section_66c": {
                "title": "Identity Theft",
                "section": "Section 66C of IT Act, 2000",
                "description": "Fraudulently or dishonestly making use of the electronic signature, password, or any other unique identification feature of any person",
                "keywords": [
                    "identity fraud", "password theft", "credential abuse", "impersonation", "electronic signature misuse",
                    "digital identity theft", "password stealing", "credential harvesting", "online identity theft",
                    "signature forgery", "identity misrepresentation", "authentication theft", "login credential theft",
                    "biometric identity theft", "digital impersonation", "unique identifier theft", "OTP theft",
                    "IT Act 66C", "electronic identity fraud", "online credential theft", "password misuse",
                    "authentication data theft", "digital signature theft", "identity credential theft",
                    "electronic identity misappropriation", "online authentication fraud"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 66C of IT Act, 2000"],
                    "punishment": "Imprisonment up to 3 years and fine up to ₹1 lakh",
                    "aggravating_factors": ["Financial loss", "Multiple victims", "Sensitive credentials"]
                }
            },
            "it_act_section_66d": {
                "title": "Cheating by Personation by using computer resource",
                "section": "Section 66D of IT Act, 2000",
                "description": "Cheating by impersonation using computer resource or communication device",
                "keywords": [
                    "impersonation", "electronic fraud", "online cheating", "fake identity", "digital impersonation",
                    "online identity fraud", "profile impersonation", "digital personation", "fake profile",
                    "fraudulent representation", "identity deception", "online impersonation fraud", "fake social media account",
                    "false persona", "identity spoofing", "pretexting", "impersonation scam", "IT Act 66D",
                    "digital deception", "fraudulent digital identity", "cheating by impersonation", "online deception",
                    "communication device fraud", "false identity fraud", "electronic impersonation", "online persona fraud"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 66D of IT Act, 2000"],
                    "punishment": "Imprisonment up to 3 years and fine up to ₹1 lakh",
                    "aggravating_factors": ["Financial fraud", "Impersonating officials", "Mass targeting"]
                }
            },
            "it_act_section_66e": {
                "title": "Violation of Privacy",
                "section": "Section 66E of IT Act, 2000",
                "description": "Capturing, publishing or transmitting images of private areas of any person without consent",
                "keywords": [
                    "privacy violation", "unauthorized photography", "private images", "consent violation", "voyeurism",
                    "image privacy", "non-consensual images", "intimate images", "privacy breach", "unauthorized recording",
                    "private area images", "privacy intrusion", "unauthorized surveillance", "hidden camera",
                    "non-consensual photography", "image capture violation", "unauthorized image sharing",
                    "IT Act 66E", "privacy violation penalty", "digital voyeurism", "image privacy breach",
                    "unauthorized image transmission", "image consent violation", "private recording",
                    "intimate privacy violation", "body privacy", "digital privacy violation"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 66E of IT Act, 2000"],
                    "punishment": "Imprisonment up to 3 years or fine up to ₹2 lakh, or both",
                    "aggravating_factors": ["Distribution of content", "Blackmail intent", "Repeated violations"]
                }
            },
            "it_act_section_66f": {
                "title": "Cyber Terrorism",
                "section": "Section 66F of IT Act, 2000",
                "description": "Acts of cybercrime with intent to threaten the unity, integrity, security or sovereignty of India",
                "keywords": [
                    "cyber terrorism", "national security threat", "critical infrastructure attack", "digital sabotage",
                    "cyber attack national security", "digital terrorism", "critical information infrastructure",
                    "cyber warfare", "nation state attack", "cyber sabotage", "strategic system attack",
                    "cybersecurity threat", "sovereign system attack", "national digital infrastructure",
                    "cybersecurity breach", "IT Act 66F", "anti-national cyber activity", "sovereignty threat",
                    "cyber integrity threat", "unity threat", "national digital security", "cyber terrorism act",
                    "critical infrastructure sabotage", "national sovereignty cyber attack", "digital insurgency"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 66F of IT Act, 2000"],
                    "punishment": "Imprisonment which may extend to life imprisonment",
                    "aggravating_factors": ["Loss of life", "Critical infrastructure damage", "Foreign involvement"]
                }
            },
            "it_act_section_67": {
                "title": "Publishing or transmitting obscene material in electronic form",
                "section": "Section 67 of IT Act, 2000",
                "description": "Publishing or transmitting material which is lascivious or appeals to prurient interest in electronic form",
                "keywords": [
                    "obscene content", "digital pornography", "electronic obscenity", "indecent material",
                    "obscene publication", "digital obscenity", "prurient content", "lascivious material",
                    "obscene electronic transmission", "online obscene content", "digital indecency",
                    "obscene digital publication", "online obscene material", "obscene transmission",
                    "IT Act 67", "electronic obscene content", "digital obscene material", "indecent digital content",
                    "electronic pornography", "obscene digital communication", "obscene electronic content",
                    "electronic indecent transmission", "online pornography", "electronic obscene material",
                    "digital indecent content"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 67 of IT Act, 2000"],
                    "punishment": "First conviction: Up to 3 years and fine up to ₹5 lakh; Subsequent conviction: Up to 5 years and fine up to ₹10 lakh",
                    "aggravating_factors": ["Commercial distribution", "Minor access", "Repeat offense"]
                }
            },
            "it_act_section_67a": {
                "title": "Publishing or transmitting of material containing sexually explicit act in electronic form",
                "section": "Section 67A of IT Act, 2000",
                "description": "Publishing or transmitting material containing sexually explicit acts in electronic form",
                "keywords": [
                    "explicit content", "sexual material", "adult content", "pornographic transmission",
                    "sexually explicit publication", "electronic sexual content", "digital sexual material",
                    "explicit electronic content", "digital explicit material", "explicit content sharing",
                    "sexual content transmission", "explicit electronic publication", "sexual act depiction",
                    "IT Act 67A", "electronic sexual material", "digital pornography", "explicit content distribution",
                    "sexual content publication", "electronic explicit material", "digital sexual content",
                    "online sexual material", "electronic pornography", "sexually explicit digital content",
                    "sexual electronic transmission", "explicit digital material"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 67A of IT Act, 2000"],
                    "punishment": "First conviction: Up to 5 years and fine up to ₹10 lakh; Subsequent conviction: Up to 7 years and fine up to ₹10 lakh",
                    "aggravating_factors": ["Commercial intent", "Wide distribution", "Non-consensual content"]
                }
            },
            "it_act_section_67b": {
                "title": "Publishing or transmitting material depicting children in sexually explicit act in electronic form",
                "section": "Section 67B of IT Act, 2000",
                "description": "Publishing, transmitting, collecting, seeking, browsing, or storing material depicting children in sexually explicit acts in electronic form",
                "keywords": [
                    "child sexual abuse material", "CSAM", "child pornography", "minor exploitation",
                    "child sexual content", "child exploitation material", "underage sexual images",
                    "child abuse imagery", "child explicit content", "child sexual imagery", "minor sexual content",
                    "underage explicit material", "child sexual exploitation", "electronic child exploitation",
                    "IT Act 67B", "child abuse material", "CSEM", "digital child exploitation", "online child abuse",
                    "child sexual abuse online", "digital CSAM", "minor sexual material", "child sexual abuse images",
                    "electronic child pornography", "child exploitation imagery", "underage sexual content"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 67B of IT Act, 2000"],
                    "punishment": "First conviction: Up to 5 years and fine up to ₹10 lakh; Subsequent conviction: Up to 7 years and fine up to ₹10 lakh",
                    "aggravating_factors": ["Direct involvement with children", "Distribution network", "Large collection"]
                }
            },
            "it_act_section_67c": {
                "title": "Preservation and retention of information by intermediaries",
                "section": "Section 67C of IT Act, 2000",
                "description": "Intermediary intentionally or knowingly contravening the directions about preservation and retention of information",
                "keywords": [
                    "data retention", "evidence preservation", "intermediary compliance", "information storage",
                    "data preservation", "intermediary obligation", "information retention", "data storage requirements",
                    "electronic records retention", "digital evidence preservation", "transaction data retention",
                    "log preservation", "record keeping obligation", "data maintenance", "IT Act 67C",
                    "data preservation failure", "information retention violation", "intermediary non-compliance",
                    "data storage violation", "record retention failure", "information preservation breach",
                    "digital record retention", "intermediary data obligation", "log retention requirement",
                    "electronic record maintenance"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 67C of IT Act, 2000"],
                    "punishment": "Imprisonment up to 3 years and fine",
                    "aggravating_factors": ["Intentional destruction", "Obstruction of investigation", "Large scale non-compliance"]
                }
            },
            "it_act_section_68": {
                "title": "Failure to comply with Controller's directions",
                "section": "Section 68 of IT Act, 2000",
                "description": "Failure to comply with the directions given by Controller of Certifying Authorities",
                "keywords": [
                    "controller directions", "non-compliance", "certification authority", "regulatory failure",
                    "controller non-compliance", "certification regulation", "authority directive failure",
                    "compliance violation", "CCA directions", "IT Act 68", "controller order violation",
                    "certifying authority non-compliance", "controller instruction disobedience", "CCA compliance",
                    "digital certification compliance", "regulatory order violation", "certificate compliance",
                    "controller direction breach", "certification directive failure", "digital certificate compliance",
                    "certification authority violation", "electronic certification regulation", "controller mandate violation"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 68 of IT Act, 2000"],
                    "punishment": "Imprisonment up to 3 years or fine up to ₹2 lakh, or both",
                    "aggravating_factors": ["Repeat violation", "Public impact", "Willful disobedience"]
                }
            },
            "it_act_section_69": {
                "title": "Failure to assist in interception or monitoring",
                "section": "Section 69 of IT Act, 2000",
                "description": "Failure to assist authorized agency with interception, monitoring, or decryption of information",
                "keywords": [
                    "interception refusal", "monitoring non-compliance", "decryption failure", "surveillance assistance",
                    "lawful interception", "monitoring assistance", "decryption assistance", "information access",
                    "interception compliance", "government surveillance assistance", "IT Act 69", "decryption order",
                    "interception order", "monitoring directive", "lawful access", "encryption assistance",
                    "interception assistance refusal", "surveillance cooperation", "lawful monitoring",
                    "technical assistance order", "decryption non-compliance", "interception non-cooperation",
                    "authorized monitoring", "information interception", "legal surveillance"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 69 of IT Act, 2000"],
                    "punishment": "Imprisonment up to 7 years and fine",
                    "aggravating_factors": ["Intentional obstruction", "National security implications", "Systematic non-compliance"]
                }
            },
            "it_act_section_69a": {
                "title": "Failure to comply with blocking orders",
                "section": "Section 69A of IT Act, 2000",
                "description": "Failure of the intermediary to comply with direction for blocking public access to information",
                "keywords": [
                    "blocking directive", "content removal", "access restriction", "intermediary compliance",
                    "website blocking", "content blocking", "access blocking", "information restriction",
                    "internet blocking", "online content removal", "IT Act 69A", "content censorship",
                    "blocking order", "web access restriction", "content filtering", "intermediary blocking",
                    "public access restriction", "blocking non-compliance", "content takedown", "information filtering",
                    "internet content removal", "digital content restriction", "website takedown", "web filtering",
                    "internet censorship"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 69A of IT Act, 2000"],
                    "punishment": "Imprisonment up to 7 years and fine",
                    "aggravating_factors": ["Willful disobedience", "Public harm", "Repeat violation"]
                }
            },
            "it_act_section_69b": {
                "title": "Failure to comply with cybersecurity monitoring directions",
                "section": "Section 69B of IT Act, 2000",
                "description": "Intermediary contravening provisions regarding monitoring and collecting traffic data for cybersecurity",
                "keywords": [
                    "traffic monitoring", "cybersecurity compliance", "data collection", "network surveillance",
                    "security monitoring", "traffic data", "cybersecurity monitoring", "network monitoring",
                    "security data collection", "traffic analysis", "cyber intelligence", "IT Act 69B",
                    "traffic inspection", "security surveillance", "network data collection", "security compliance",
                    "intermediary monitoring", "traffic data retention", "cyber traffic analysis", "security intelligence",
                    "digital surveillance", "network traffic monitoring", "cyber data collection", "security traffic data",
                    "cybersecurity intelligence"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 69B of IT Act, 2000"],
                    "punishment": "Imprisonment up to 3 years and fine",
                    "aggravating_factors": ["Critical infrastructure", "Persistent non-compliance", "Data destruction"]
                }
            },
            "it_act_section_70": {
                "title": "Unauthorized access to protected systems",
                "section": "Section 70 of IT Act, 2000",
                "description": "Securing or attempting to secure access to protected system declared by government",
                "keywords": [
                    "critical infrastructure", "protected system", "unauthorized access", "secured computer",
                    "critical system access", "government protected computer", "notified system", "critical network",
                    "protected infrastructure", "national system", "security breach", "IT Act 70", "critical IT system",
                    "sensitive system access", "protected computer access", "government computer security",
                    "critical digital infrastructure", "secured system breach", "important computer system",
                    "notified protected system", "essential computer facility", "protected network access",
                    "nationally important system", "critical computer resource", "government protected network"
                ],
                "legal_framework": {
                    "primary_sections": ["Section 70 of IT Act, 2000"],
                    "punishment": "Imprisonment up to 10 years and fine",
                    "aggravating_factors": ["Data theft", "System damage", "National security impact"]
                }
            },
           
            # For section 70B (CERT-In)
"it_act_section_70b": {
    "title": "Failure to provide information to CERT-In",
    "section": "Section 70B of IT Act, 2000",
    "description": "Failure to provide information called for or comply with direction issued by CERT-In",
    "keywords": [
        "CERT-In", "incident reporting", "cybersecurity compliance", "information sharing",
        "security incident", "cyber incident notification", "mandatory reporting", "security breach reporting",
        "CERT-In directives", "incident response", "cybersecurity coordination", "computer emergency response",
        "security incident disclosure", "breach notification", "national cybersecurity", "IT Act 70B",
        "cyber incident compliance", "security information sharing", "incident notification requirement",
        "security response coordination", "cyber emergency reporting", "digital incident response",
        "cybersecurity cooperation", "security breach notification", "incident handling procedures"
    ],
    "legal_framework": {
        "primary_sections": ["Section 70B of IT Act, 2000"],
        "punishment": "Imprisonment up to 1 year or fine up to ₹1 lakh, or both",
        "aggravating_factors": ["Critical incident", "Delayed notification", "Concealment of breach"]
    }
},

# For section 71 (Misrepresentation)
"it_act_section_71": {
    "title": "Misrepresentation to Controller or Certifying Authority",
    "section": "Section 71 of IT Act, 2000",
    "description": "Making false statements or representation to Controller of Certifying Authorities or Certifying Authority",
    "keywords": [
        "false representation", "certification fraud", "digital signature misrepresentation", "authority deception",
        "certificate falsehood", "false statements", "misrepresentation offense", "certification authority deception",
        "controller misrepresentation", "digital certificate fraud", "certification falsehood", "authentication fraud",
        "false certification information", "certificate authority deception", "IT Act 71", "false digital certificate", 
        "signature authority fraud", "certification misrepresentation", "digital identity fraud", "certificate misstatement",
        "controller deception", "certification document fraud", "digital certificate misrepresentation",
        "signature verification fraud", "authentication authority deception"
    ],
    "legal_framework": {
        "primary_sections": ["Section 71 of IT Act, 2000"],
        "punishment": "Imprisonment up to 3 years or fine up to ₹1 lakh, or both",
        "aggravating_factors": ["Fraudulent intent", "Public impact", "Multiple misrepresentations"]
    }
},

# For section 72 (Confidentiality breach)
"it_act_section_72": {
    "title": "Breach of Confidentiality and Privacy",
    "section": "Section 72 of IT Act, 2000",
    "description": "Disclosure of confidential information by person with official access without consent",
    "keywords": [
        "confidentiality breach", "unauthorized disclosure", "privacy violation", "information leak",
        "confidential data exposure", "official secrecy breach", "authorized access misuse", "privileged information disclosure",
        "data confidentiality violation", "unauthorized information sharing", "official trust breach", "privacy breach",
        "IT Act 72", "access privilege abuse", "confidentiality violation", "information security breach",
        "secure data disclosure", "unauthorized revelation", "secret information exposure", "professional confidence breach",
        "data privacy violation", "information protection failure", "confidential knowledge disclosure",
        "authorized personnel breach", "secure information leak"
    ],
    "legal_framework": {
        "primary_sections": ["Section 72 of IT Act, 2000"],
        "punishment": "Imprisonment up to 2 years or fine up to ₹1 lakh, or both",
        "aggravating_factors": ["Sensitive information", "Multiple breaches", "Commercial intent"]
    }
},

# For section 72A (Contract breach)
"it_act_section_72a": {
    "title": "Disclosure of information in breach of lawful contract",
    "section": "Section 72A of IT Act, 2000",
    "description": "Disclosure of personal information in breach of lawful contract by service provider",
    "keywords": [
        "contract violation", "data protection breach", "confidentiality agreement", "personal data disclosure",
        "service provider breach", "contractual confidentiality", "data protection violation", "personal information misuse",
        "confidentiality contract breach", "service agreement violation", "customer data exposure", "contractual obligation breach",
        "IT Act 72A", "lawful contract violation", "client data misuse", "confidential information sharing",
        "data handling breach", "provider trust violation", "sensitive information disclosure", "customer privacy breach",
        "data custody violation", "terms of service breach", "personal data protection failure", "provider confidentiality violation",
        "data stewardship breach", "contractual privacy violation"
    ],
    "legal_framework": {
        "primary_sections": ["Section 72A of IT Act, 2000"],
        "punishment": "Imprisonment up to 3 years or fine up to ₹5 lakh, or both",
        "aggravating_factors": ["Commercial advantage", "Sensitive personal data", "Large scale disclosure"]
    }
},

# For section 73 (False Digital Signature Certificate)
"it_act_section_73": {
    "title": "Publishing false Digital Signature Certificate",
    "section": "Section 73 of IT Act, 2000",
    "description": "Publishing electronic Signature Certificate false in certain particulars",
    "keywords": [
        "false certificate", "digital signature fraud", "certificate forgery", "electronic authentication",
        "fraudulent digital certificate", "signature certificate falsification", "PKI fraud", "certificate authenticity",
        "digital identity forgery", "electronic certificate fraud", "signature verification fraud", "authentication certificate fraud",
        "IT Act 73", "digital certificate misrepresentation", "certificate integrity violation", "electronic signature fraud",
        "digital certificate forgery", "authentication credential fraud", "false verification certificate", "identity certificate fraud",
        "electronic signature certificate", "digital authentication fraud", "certificate publication offense",
        "electronic verification fraud", "identity verification certificate"
    ],
    "legal_framework": {
        "primary_sections": ["Section 73 of IT Act, 2000"],
        "punishment": "Imprisonment up to 2 years or fine up to ₹1 lakh, or both",
        "aggravating_factors": ["Multiple certificates", "Commercial gain", "Identity theft"]
    }
},

# For section 74 (Fraudulent Publication)
"it_act_section_74": {
    "title": "Publication for fraudulent purpose",
    "section": "Section 74 of IT Act, 2000",
    "description": "Publication of Digital Signature Certificate for fraudulent purpose",
    "keywords": [
        "fraudulent certificate", "deceptive publication", "certificate misuse", "digital fraud",
        "certificate fraud purpose", "signature certificate scam", "fraudulent digital identity", "malicious certificate publication",
        "digital certificate misuse", "signature fraud", "deceptive digital certificate", "certificate publication fraud",
        "IT Act 74", "fraudulent verification certificate", "deceptive authentication", "signature identity fraud",
        "certificate scheme", "electronic identity fraud", "signature certificate deception", "fraudulent digital verification",
        "electronic certificate scam", "authentication fraud", "digital identity misrepresentation", "fraudulent electronic certificate",
        "digital certificate scheme"
    ],
    "legal_framework": {
        "primary_sections": ["Section 74 of IT Act, 2000"],
        "punishment": "Imprisonment up to 2 years or fine up to ₹1 lakh, or both",
        "aggravating_factors": ["Financial fraud", "Multiple victims", "Systematic operation"]
    }
}
        },
        "intents": [
    {
        "tag": "greeting",
        "patterns": [
            "hi", "hello", "hey", "greetings", "good morning", "good evening", "good afternoon",
            "hi there", "hello there", "howdy", "what's up", "namaste", "hola", "start", 
            "begin conversation", "hey assistant", "hi bot", "hello cyber assistant",
            "cyber law assistant", "let's talk", "hi cyber law expert", "help me"
        ],
        "responses": [
            "Hello! How can I help you with cyber law questions today?",
            "Hi there! What cyber law information do you need?",
            "Greetings! I'm your cyber law assistant. What would you like to know?",
            "Welcome! I'm here to help with your questions about Indian IT laws and cyber security.",
            "Hello! I'm specialized in Indian cyber laws. How may I assist you today?",
            "Hi! Ask me about IT Act sections, cyber crimes, or reporting procedures in India."
        ]
    },
    {
        "tag": "law_information",
        "patterns": [
            "tell me about", "what is", "explain", "describe", "details on", "information about",
            "define", "elaborate on", "clarify", "what does mean", "information regarding",
            "meaning of", "tell me more about", "what are", "how does work", "concept of",
            "details about", "information on", "summarize", "brief me on", "overview of",
            "enlighten me about", "educate me on", "what exactly is", "can you explain"
        ],
        "responses": [
            "Here's what I know about {topic}...",
            "Let me provide information about {topic}...",
            "According to cyber laws, {topic} refers to...",
            "In the context of Indian IT laws, {topic} is defined as...",
            "The IT Act describes {topic} as...",
            "Under Indian cyber legislation, {topic} encompasses...",
            "Let me explain the concept of {topic} according to the IT Act...",
            "Here's a comprehensive explanation of {topic} under Indian cyber laws..."
        ]
    },
    {
        "tag": "reporting_procedure",
        "patterns": [
            "how to report", "reporting process", "file complaint", "who to contact", "where to report",
            "steps to report", "complaint procedure", "report cyber crime", "lodge complaint",
            "report incident", "cybercrime reporting", "file FIR", "notify authorities",
            "reporting channel", "official complaint", "report to police", "cybercrime portal",
            "online reporting", "complaint mechanism", "inform authorities", "legal procedure for reporting",
            "proper channel to report", "official reporting", "register complaint", "notify about breach"
        ],
        "responses": [
            "To report {crime}, you should follow these steps...",
            "The reporting procedure for {crime} includes...",
            "For reporting {crime}, the recommended approach is...",
            "To file a complaint about {crime}, here's the official process...",
            "The standard procedure for reporting {crime} in India involves...",
            "If you've experienced {crime}, here's how to properly report it to authorities...",
            "For {crime} incidents, follow this reporting protocol for proper legal action...",
            "The most effective way to report {crime} is through these official channels..."
        ]
    },
    {
        "tag": "prevention",
        "patterns": [
            "how to prevent", "protection against", "safeguard", "security measures", "avoid", "protect from",
            "preventive steps", "safety measures", "mitigate risk", "defensive measures", "safeguarding against",
            "securing against", "protective steps", "avoid being victim", "reduce vulnerability",
            "preventive action", "risk reduction", "secure from", "safety protocols", "cybersecurity measures",
            "best practices", "security protocols", "prevention tips", "safety guidelines", "precautions against"
        ],
        "responses": [
            "To prevent {crime}, consider these measures...",
            "Protection against {crime} includes...",
            "Here are some prevention tips for {crime}...",
            "Safeguarding yourself from {crime} requires...",
            "To minimize the risk of {crime}, implement these security practices...",
            "Effective protection against {crime} involves these key strategies...",
            "Security experts recommend these measures to prevent {crime}...",
            "To protect yourself or your organization from {crime}, follow these best practices..."
        ]
    },
    {
        "tag": "legal_consequences",
        "patterns": [
            "punishment", "penalty", "sentence", "fine", "imprisonment", "consequences",
            "legal action", "prosecution", "criminal charges", "sanctions", "legal penalties",
            "jail time", "criminal punishment", "legal ramifications", "court action",
            "punitive measures", "legal repercussions", "statutory penalties", "conviction consequences",
            "judicial punishment", "legal liability", "punitive action", "what happens if caught",
            "punishment under law", "legal penalties for offense"
        ],
        "responses": [
            "The legal consequences for {crime} include...",
            "Under the IT Act, {crime} is punishable by...",
            "For {crime}, the penalties can range from...",
            "Indian law prescribes the following punishment for {crime}...",
            "If convicted of {crime}, an individual may face...",
            "The IT Act stipulates these specific penalties for {crime}...",
            "Legal consequences for {crime} under Indian cyber law include...",
            "The judiciary typically imposes these punishments for {crime} cases..."
        ]
    },
    {
        "tag": "evidence_collection",
        "patterns": [
            "evidence needed", "prove", "documentation", "collect evidence", "document incident",
            "types of evidence", "digital evidence", "forensic requirements", "what evidence",
            "documenting breach", "proof required", "evidence gathering", "forensic collection",
            "legal evidence", "documentation procedure", "evidence preservation", "case building",
            "admissible evidence", "court proof", "prosecutorial evidence", "evidential requirements",
            "evidence trail", "chain of custody", "digital forensics", "incident documentation"
        ],
        "responses": [
            "To document a {crime} incident, collect the following evidence...",
            "Important evidence for {crime} cases includes...",
            "For prosecuting {crime}, you should gather these types of evidence...",
            "Effective documentation of {crime} requires these specific records...",
            "Digital forensics experts recommend collecting this evidence for {crime} cases...",
            "To build a strong legal case for {crime}, ensure you preserve...",
            "The chain of evidence for {crime} should include these critical elements...",
            "Courts typically require these forms of evidence when prosecuting {crime}..."
        ]
    },
    {
        "tag": "section_inquiry",
        "patterns": [
            "section", "IT Act section", "what does section", "explain section", "provision of", "law section",
            "section details", "legal provision", "act section", "statutory provision", "legal section",
            "section number", "IT legislation", "law provision", "statute section", "clause in act",
            "legal clause", "section in IT Act", "what is section about", "provision details",
            "law clause", "section meaning", "provision explanation", "section interpretation", "legislative provision"
        ],
        "responses": [
            "Section {section} of the IT Act pertains to {description}...",
            "As per the IT Act, Section {section} covers {description}...",
            "The IT Act's Section {section} deals with {description}...",
            "Section {section} specifically addresses {description} under Indian cyber law...",
            "The legal provision in Section {section} of the IT Act states that {description}...",
            "Under Indian cyber legislation, Section {section} is concerned with {description}...",
            "The IT Act Section {section} establishes legal framework for {description}...",
            "Section {section}'s primary focus is on {description} with specific provisions for..."
        ]
    },
    {
        "tag": "case_law",
        "patterns": [
            "landmark case", "precedent", "court ruling", "legal case", "judicial decision",
            "famous case", "court precedent", "case study", "legal judgment", "important decision",
            "high court ruling", "supreme court case", "judicial precedent", "significant case",
            "case law", "legal history", "court decision", "judicial history", "case example",
            "legal interpretation", "court judgment", "case verdict", "notable case", "legal reference case"
        ],
        "responses": [
            "A landmark case regarding {topic} is {case_name}, which established...",
            "The judicial precedent for {topic} was set in {case_name}, where the court ruled...",
            "In the case of {case_name}, the court's interpretation of {topic} established that...",
            "Legal understanding of {topic} was significantly shaped by {case_name}, in which...",
            "The {case_name} case provides important guidance on {topic}, particularly regarding...",
            "Courts typically refer to {case_name} when adjudicating matters related to {topic}...",
            "The legal framework for {topic} was tested in {case_name}, resulting in..."
        ]
    },
    {
        "tag": "jurisdiction",
        "patterns": [
            "jurisdiction", "which court", "legal authority", "territorial application",
            "which police", "jurisdictional issues", "cyber jurisdiction", "legal territory",
            "cross-border", "international jurisdiction", "court authority", "territorial reach",
            "applicable law", "legal boundary", "jurisdictional challenge", "enforcement jurisdiction",
            "territorial jurisdiction", "legal reach", "jurisdiction determination", "authority limits",
            "jurisdiction transfer", "multi-state jurisdiction", "international case", "local enforcement"
        ],
        "responses": [
            "For {crime} cases, jurisdiction is typically determined by...",
            "The jurisdictional considerations for {crime} under Indian law include...",
            "When {crime} crosses borders, the jurisdiction is established based on...",
            "The IT Act addresses jurisdictional challenges for {crime} by...",
            "In cases of {crime}, the following factors determine proper jurisdiction...",
            "Jurisdiction for online {crime} is complex and involves these key principles...",
            "Indian courts establish jurisdiction over {crime} cases through these legal mechanisms..."
        ]
    },
    {
        "tag": "recent_updates",
        "patterns": [
            "recent changes", "new amendments", "latest updates", "law updates", "recent modifications",
            "new provisions", "act amendments", "current laws", "updated regulations", "legal revisions",
            "recent developments", "latest legislation", "new rules", "regulatory updates", "law revisions",
            "policy changes", "new guidelines", "recent legal developments", "legislation updates",
            "contemporary laws", "modern provisions", "updated legal framework", "current legal status"
        ],
        "responses": [
            "Recent updates to laws concerning {topic} include...",
            "The IT Act provisions related to {topic} were recently amended to...",
            "As of our last update, the legal framework for {topic} has changed in these ways...",
            "Current legislation around {topic} has evolved to address...",
            "The most recent amendments affecting {topic} have introduced...",
            "Legal approaches to {topic} have been modernized through these recent changes...",
            "Regulatory updates for {topic} now include these important provisions..."
        ]
    },
    {
        "tag": "thank_you",
        "patterns": [
            "thanks", "thank you", "appreciate it", "helpful", "thank you so much", "thanks a lot",
            "very helpful", "that helps", "appreciate your help", "great help", "thanks for info",
            "gratitude", "much appreciated", "thank you for assistance", "thanks for explaining",
            "that was useful", "good information", "thanks for clarification", "well explained",
            "thanks for your time", "appreciate the guidance", "thanks for the details"
        ],
        "responses": [
            "You're welcome! Feel free to ask if you have more questions about cyber law.",
            "Happy to help! Let me know if you need more information.",
            "You're welcome. Is there anything else you'd like to know about cyber laws?",
            "Glad I could assist! Don't hesitate to ask about other IT law topics.",
            "My pleasure! I'm here for any other cyber law questions you might have.",
            "You're welcome! Understanding cyber laws is important, so feel free to ask more.",
            "Anytime! Let me know if you need clarification on any other cyber law matters."
        ]
    },
    {
        "tag": "goodbye",
        "patterns": [
            "bye", "goodbye", "see you", "that's all", "end", "quit", "exit", "good night",
            "farewell", "have a good day", "until next time", "closing", "terminate", "finish",
            "end conversation", "that will be all", "no more questions", "signing off",
            "done for now", "thanks bye", "leaving now", "conversation over", "that's it"
        ],
        "responses": [
            "Goodbye! Feel free to return if you have more cyber law questions.",
            "Take care! I'm here if you need cyber law assistance in the future.",
            "Goodbye and stay cyber-safe!",
            "Until next time! Remember to practice good cyber security.",
            "Farewell! If you have more questions about IT laws, I'll be here.",
            "Goodbye! Stay informed about cyber laws to protect yourself online.",
            "Take care and be safe in your digital activities!"
        ]
    },
    {
        "tag": "data_protection",
        "patterns": [
            "data protection", "information security", "privacy law", "personal data", "data privacy",
            "data security", "information protection", "private information", "confidential data",
            "data protection law", "privacy regulation", "personal information protection", "data rights",
            "data subject rights", "privacy compliance", "data protection framework", "privacy legislation",
            "data security practices", "information privacy", "data handling", "privacy requirements",
            "digital privacy", "protect personal data", "information safety", "cybersecurity compliance"
        ],
        "responses": [
            "Data protection under Indian law includes these key principles...",
            "For personal data protection, the IT Act provides these safeguards...",
            "The legal framework for data privacy in India consists of...",
            "Organizations handling personal data must comply with these requirements...",
            "Under current Indian law, data protection encompasses these rights and obligations...",
            "The data protection regime in India addresses privacy through these mechanisms...",
            "Personal information protection is governed by these specific provisions..."
        ]
    },
    {
        "tag": "intermediary_liability",
        "patterns": [
            "intermediary liability", "platform responsibility", "social media liability", "website responsibility",
            "online platform duty", "intermediary obligations", "hosting liability", "internet provider responsibility",
            "social network liability", "online intermediary", "safe harbor provisions", "platform legal obligations",
            "internet intermediary", "website legal responsibility", "app liability", "digital platform responsibility",
            "e-commerce liability", "online marketplace responsibility", "content host liability", "due diligence requirements",
            "takedown obligation", "ISP liability", "content removal duty", "intermediary guidelines"
        ],
        "responses": [
            "Intermediary liability under Indian law is governed by...",
            "Online platforms have these specific legal responsibilities in India...",
            "The IT Act establishes these requirements for intermediaries...",
            "Social media and other online platforms must follow these guidelines...",
            "The legal framework for intermediary liability includes these key obligations...",
            "Under Indian cyber law, intermediaries are protected if they comply with...",
            "Content hosting platforms must follow these due diligence requirements..."
        ]
    }
]
    }
    
        return knowledge_base
    def build_vector_index(self):
        """Build vector representations of knowledge base items for semantic search"""
        vector_index = {}
        
        if not self.nlp_engine or not hasattr(self.nlp_engine, "get_embeddings"):
            logger.warning("NLP engine not available for vector indexing")
            return vector_index
        
        try:
            # Create embeddings for laws
            for law_id, law_info in self.knowledge_base.get("laws", {}).items():
                # Create a document text combining important fields for better semantic matching
                doc_text = f"{law_info.get('title', '')}. {law_info.get('description', '')}. " + \
                          f"Keywords: {', '.join(law_info.get('keywords', []))}."
                
                # Generate embedding
                embedding = self.nlp_engine.get_embeddings(doc_text)
                vector_index[f"law:{law_id}"] = {
                    "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    "type": "law",
                    "id": law_id
                }
                
            # Create embeddings for intents
            for intent in self.knowledge_base.get("intents", []):
                # Combine patterns for better semantic matching
                doc_text = f"{intent.get('tag', '')}. {'. '.join(intent.get('patterns', []))}"
                
                # Generate embedding
                embedding = self.nlp_engine.get_embeddings(doc_text)
                vector_index[f"intent:{intent.get('tag')}"] = {
                    "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    "type": "intent",
                    "id": intent.get('tag')
                }
                
            logger.info(f"Built vector index with {len(vector_index)} entries")
            return vector_index
            
        except Exception as e:
            logger.error(f"Error building vector index: {e}")
            return {}
    
    def search(self, query, top_n=3):
        """Search the knowledge base semantically and by keywords"""
        results = []
        
        # Try semantic search first if NLP engine is available
        if self.nlp_engine and hasattr(self.nlp_engine, "get_embeddings"):
            semantic_results = self.semantic_search(query, top_n)
            results.extend(semantic_results)
        
        # Fall back to keyword search if semantic search fails or produces no results
        if not results:
            keyword_results = self.keyword_search(query, top_n)
            results.extend(keyword_results)
            
        return results
    
    def semantic_search(self, query, top_n=3):
        """Search using vector representations and semantic similarity"""
        if not self.nlp_engine or not self.vector_index:
            return []
            
        try:
            # Generate embedding for query
            query_embedding = self.nlp_engine.get_embeddings(query)
            if not isinstance(query_embedding, np.ndarray):
                return []
                
            # Calculate similarity with all items in vector index
            similarities = []
            for key, item in self.vector_index.items():
                if not item.get("embedding"):
                    continue
                    
                item_embedding = np.array(item["embedding"])
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    item_embedding.reshape(1, -1)
                )[0][0]
                
                similarities.append((key, similarity, item))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top N results
            results = []
            for key, similarity, item in similarities[:top_n]:
                if similarity < 0.3:  # Threshold for meaningful results
                    continue
                    
                if item["type"] == "law":
                    law_id = item["id"]
                    law_info = self.knowledge_base["laws"].get(law_id, {})
                    results.append({
                        "type": "law",
                        "id": law_id,
                        "content": law_info,
                        "similarity": similarity,
                        "relevance": "high" if similarity > 0.7 else "medium" if similarity > 0.5 else "low"
                    })
                elif item["type"] == "intent":
                    intent_tag = item["id"]
                    intent_info = next((i for i in self.knowledge_base["intents"] if i.get("tag") == intent_tag), {})
                    results.append({
                        "type": "intent",
                        "id": intent_tag,
                        "content": intent_info,
                        "similarity": similarity,
                        "relevance": "high" if similarity > 0.7 else "medium" if similarity > 0.5 else "low"
                    })
                    
            return results
                
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def keyword_search(self, query, top_n=3):
        """Search using keyword matching as fallback"""
        results = []
        query_lower = query.lower()
        
        # Search laws by keywords
        for law_id, law_info in self.knowledge_base.get("laws", {}).items():
            matches = 0
            
            # Check title
            if law_info.get("title", "").lower() in query_lower:
                matches += 10
                
            # Check description
            if any(word in query_lower for word in law_info.get("description", "").lower().split()):
                matches += 5
                
            # Check keywords
            for keyword in law_info.get("keywords", []):
                if keyword.lower() in query_lower:
                    matches += 8
                    
            if matches > 0:
                results.append({
                    "type": "law",
                    "id": law_id,
                    "content": law_info,
                    "similarity": min(matches / 20, 1.0),  # Normalize to 0-1 range
                    "relevance": "high" if matches > 15 else "medium" if matches > 7 else "low"
                })
        
        # Search intents by patterns
        for intent in self.knowledge_base.get("intents", []):
            matches = 0
            
            # Check patterns
            for pattern in intent.get("patterns", []):
                if pattern.lower() in query_lower:
                    matches += 10
                elif any(word in query_lower for word in pattern.lower().split()):
                    matches += 3
                    
            if matches > 0:
                results.append({
                    "type": "intent",
                    "id": intent.get("tag"),
                    "content": intent,
                    "similarity": min(matches / 20, 1.0),  # Normalize to 0-1 range
                    "relevance": "high" if matches > 15 else "medium" if matches > 7 else "low"
                })
                
        # Sort results by similarity (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top N results
        return results[:top_n]
    
    def get_law_by_id(self, law_id):
        """Get law details by ID"""
        return self.knowledge_base.get("laws", {}).get(law_id)
    
    def get_intent_by_tag(self, tag):
        """Get intent details by tag"""
        for intent in self.knowledge_base.get("intents", []):
            if intent.get("tag") == tag:
                return intent
        return None
    
    def save_knowledge_base(self):
        """Save the current knowledge base to file"""
        try:
            with open(self.knowledge_base_path, 'w') as file:
                json.dump(self.knowledge_base, file, indent=4)
            logger.info(f"Knowledge base saved to {self.knowledge_base_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False
    
    def update_knowledge_base(self, new_data):
        """Update the knowledge base with new data"""
        try:
            if "laws" in new_data:
                for law_id, law_info in new_data["laws"].items():
                    self.knowledge_base.setdefault("laws", {})[law_id] = law_info
                    
            if "intents" in new_data:
                # Replace intents with matching tags
                existing_tags = [intent.get("tag") for intent in self.knowledge_base.get("intents", [])]
                for new_intent in new_data["intents"]:
                    if new_intent.get("tag") in existing_tags:
                        # Replace existing intent
                        for i, intent in enumerate(self.knowledge_base.get("intents", [])):
                            if intent.get("tag") == new_intent.get("tag"):
                                self.knowledge_base["intents"][i] = new_intent
                    else:
                        # Add new intent
                        self.knowledge_base.setdefault("intents", []).append(new_intent)
            
            # Rebuild vector index with updated knowledge
            self.vector_index = self.build_vector_index()
            
            # Save updated knowledge base
            self.save_knowledge_base()
            
            return True
        except Exception as e:
            logger.error(f"Error updating knowledge base: {e}")
            return False


class ResponseGenerator:
    """Generate contextually relevant responses based on knowledge base and conversation context"""
    
    def __init__(self, knowledge_base, nlp_engine):
        self.knowledge_base = knowledge_base
        self.nlp_engine = nlp_engine
        self.templates = self._load_response_templates()
    
    def _load_response_templates(self):
        """Load response templates for different scenarios"""
        return {
            "greeting": [
                "Hello! I'm your cyber law assistant. How can I help you today?",
                "Welcome! I'm here to provide information about cyber laws. What would you like to know?",
                "Greetings! I can help with cyber law questions, reporting procedures, or prevention tips. What information do you need?"
            ],
            "goodbye": [
                "Thank you for using the cyber law assistant. Stay cyber-safe!",
                "Goodbye! Feel free to return if you have more cyber law questions.",
                "Thanks for chatting. Remember to stay vigilant online!"
            ],
            "not_understood": [
                "I'm not sure I understand your question about cyber laws. Could you rephrase it?",
                "I didn't quite catch that. Can you provide more details about your cyber law query?",
                "I'm having trouble understanding your question. Could you ask it differently or provide more context?"
            ],
            "law_information": [
                "Here's what you should know about {law_title}: {law_description} {legal_framework}",
                "Regarding {law_title}: {law_description} According to {legal_framework}",
                "{law_title} refers to {law_description} Under current laws, {legal_framework}"
            ],
            "prevention_tips": [
                "To protect against {law_title}, consider these prevention measures: {prevention_tips}",
                "Here are some recommended ways to prevent {law_title}: {prevention_tips}",
                "To minimize the risk of {law_title}, experts recommend: {prevention_tips}"
            ],
            "reporting_procedure": [
                "If you need to report a {law_title} incident, follow these steps: {reporting_procedure}",
                "The proper procedure for reporting {law_title} includes: {reporting_procedure}",
                "To report {law_title}, here's what you should do: {reporting_procedure}"
            ],
            "evidence_collection": [
                "When documenting a {law_title} incident, collect these types of evidence: {evidence_collection}",
                "For {law_title} cases, the following evidence is crucial: {evidence_collection}",
                "To build a strong case for {law_title}, gather this evidence: {evidence_collection}"
            ]
        }
    
    def generate_response(self, user_input, conversation_manager):
        """Generate a comprehensive response based on user input and conversation context"""
        if not user_input or not isinstance(user_input, str):
            return self._get_template_response("not_understood")
            
        # Preprocess user input
        if self.nlp_engine:
            processed_input = self.nlp_engine.preprocess_text(user_input)
        else:
            processed_input = user_input
            
        # Get conversation context
        context = conversation_manager.context
        conversation_phase = context.get("conversation_phase", "greeting")
        
        # Check for greetings or goodbyes first
        if conversation_phase == "greeting" or any(greeting in processed_input.lower() for greeting in ["hello", "hi", "hey", "greetings"]):
            # Update conversation phase
            conversation_manager.update_context("conversation_phase", "initial_query")
            return self._get_template_response("greeting")
            
        if any(goodbye in processed_input.lower() for goodbye in ["bye", "goodbye", "exit", "quit", "end"]):
            return self._get_template_response("goodbye")
            
        # Search knowledge base for relevant information
        search_results = self.knowledge_base.search(processed_input)
        
        # If no results found, check for similar past interactions
        if not search_results and self.nlp_engine:
            similar_interactions = conversation_manager.find_similar_interactions(
                processed_input, self.nlp_engine, top_n=1
            )
            
            if similar_interactions and similar_interactions[0]["similarity"] > 0.7:
                return similar_interactions[0]["assistant_response"]
            else:
                return self._get_template_response("not_understood")
                
        # Process search results and generate response
        response = self._process_search_results(search_results, processed_input, context)
        
        # Update conversation context based on response
        if search_results:
            top_result = search_results[0]
            if top_result["type"] == "law":
                conversation_manager.update_context("last_law_category", top_result["id"])
                conversation_manager.update_context("current_topic", top_result["content"].get("title"))
            conversation_manager.update_context("conversation_phase", "specific_guidance")
            
        return response
    
    def _process_search_results(self, search_results, user_input, context):
        """Process search results and construct a response"""
        if not search_results:
            return self._get_template_response("not_understood")
            
        # Analyze top result
        top_result = search_results[0]
        
        if top_result["type"] == "intent":
            # Handle intent-based response
            intent_tag = top_result["id"]
            intent_info = top_result["content"]
            
            if intent_tag == "greeting":
                return self._get_template_response("greeting")
            elif intent_tag == "goodbye":
                return self._get_template_response("goodbye")
            elif intent_tag == "thank_you":
                return random.choice(intent_info.get("responses", ["You're welcome!"]))
            else:
                # For other intents, check if we need law-specific information
                if context.get("last_law_category"):
                    law_info = self.knowledge_base.get_law_by_id(context["last_law_category"])
                    if law_info:
                        return self._generate_law_specific_response(intent_tag, law_info)
                
                # Generic response if no specific law context
                return random.choice(intent_info.get("responses", ["I understand you're asking about cyber laws."]))
                
        elif top_result["type"] == "law":
            # Handle law-based response
            law_id = top_result["id"]
            law_info = top_result["content"]
            
            # Determine specific aspect of law the user is asking about
            aspect = self._determine_law_aspect(user_input)
            return self._generate_law_specific_response(aspect, law_info)
            
        return self._get_template_response("not_understood")
    
    def _determine_law_aspect(self, user_input):
        """Determine which aspect of a law the user is interested in"""
        user_input_lower = user_input.lower()
        
        aspect_keywords = {
            "law_information": ["what is", "explain", "describe", "tell me about", "define", "meaning of"],
            "legal_consequences": ["punishment", "penalty", "jail", "fine", "sentence", "consequences"],
            "reporting_procedure": ["report", "complain", "file", "inform", "notify", "contact", "procedure"],
            "prevention_tips": ["prevent", "protect", "avoid", "secure", "safeguard", "security", "safety"],
            "evidence_collection": ["evidence", "proof", "document", "record", "collect", "prove"]
        }
        
        # Check for aspect keywords in user input
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return aspect
                
        # Default to general information
        return "law_information"
    
    def _generate_law_specific_response(self, aspect, law_info):
        """Generate a response for a specific aspect of a law"""
        if not law_info:
            return self._get_template_response("not_understood")
            
        template = self._get_template_response(aspect if aspect in self.templates else "law_information")
        
        # Prepare replacement values
        replacements = {
            "law_title": law_info.get("title", "this cyber crime"),
            "law_description": law_info.get("description", ""),
            "legal_framework": self._format_legal_framework(law_info.get("legal_framework", {})),
            "prevention_tips": self._format_list(law_info.get("prevention_tips", [])),
            "reporting_procedure": self._format_reporting_procedure(law_info.get("reporting_procedure", {})),
            "evidence_collection": self._format_list(law_info.get("evidence_collection", []))
        }
        
        # Replace placeholders in template
        for key, value in replacements.items():
            template = template.replace("{" + key + "}", value)
            
        # Add additional information based on aspect
        if aspect == "law_information":
            if law_info.get("landmark_cases"):
                template += f"\n\nNotable cases: {self._format_landmark_cases(law_info['landmark_cases'])}"
                
        return template
    
    def _format_legal_framework(self, framework):
        """Format legal framework information"""
        if not framework:
            return ""
            
        result = ""
        if framework.get("primary_sections"):
            result += f" It falls under {', '.join(framework['primary_sections'])}."
        if framework.get("punishment"):
            result += f" Punishment includes {framework['punishment']}."
            
        return result
    
    def _format_list(self, items):
        """Format a list of items for response"""
        if not items:
            return ""
            
        return ", ".join(items[:3]) + (f", and {len(items) - 3} more" if len(items) > 3 else "")
    
    def _format_reporting_procedure(self, procedure):
        """Format reporting procedure for response"""
        if not procedure:
            return ""
            
        individual = procedure.get("individual", [])
        if individual:
            return ", ".join(individual[:3]) + (f", and {len(individual) - 3} more steps" if len(individual) > 3 else "")
        return ""
    
    def _format_landmark_cases(self, cases):
        """Format landmark cases for response"""
        if not cases or len(cases) == 0:
            return ""
            
        case = cases[0]  # Take the first case
        return f"{case.get('name', '')}: {case.get('significance', '')}"
    
    def _get_template_response(self, template_key):
        """Get a random response from templates"""
        templates = self.templates.get(template_key, ["I understand you're asking about cyber laws."])
        return random.choice(templates)
class ScenarioAnalyzer:
    """Analyze cyber crime scenarios and map them to relevant legal sections"""
    
    def __init__(self, knowledge_base, nlp_engine):
        self.knowledge_base = knowledge_base
        self.nlp_engine = nlp_engine
        self.crime_patterns = self._load_crime_patterns()
        
    def _load_crime_patterns(self):
        """Load patterns for recognizing common elements in cyber crime scenarios"""
        return {
            "unauthorized_access": [
                "hack", "broke into", "unauthorized access", "illegal access", "gain access", 
                "breach", "compromise", "infiltrate", "login without permission"
            ],
            "data_theft": [
                "stole data", "took information", "downloaded files", "exfiltrated", "copied data",
                "data theft", "credential theft", "stole credentials", "identity theft"
            ],
            "malware": [
                "malware", "virus", "ransomware", "trojan", "worm", "spyware", "keylogger",
                "infected", "malicious software", "malicious code"
            ],
            "denial_of_service": [
                "dos attack", "ddos", "denial of service", "crashed server", "flooded",
                "made unavailable", "disrupted service"
            ],
            "fraud": [
                "fraud", "deceived", "impersonated", "fake", "pretended to be", "scam", 
                "fraudulent", "misrepresented", "social engineering"
            ],
            "harassment": [
                "harass", "stalk", "threaten", "blackmail", "extort", "intimidate",
                "bullying", "threatening messages", "defame"
            ],
            "sexual_offenses": [
                "obscene", "explicit content", "child exploitation", "revenge porn",
                "non-consensual", "intimate images", "sexually explicit"
            ],
            "intellectual_property": [
                "copyright", "piracy", "illegal download", "counterfeit", "plagiarize",
                "intellectual property", "trademark", "patent"
            ]
        }
        
    def analyze_scenario(self, scenario_text):
        """Analyze a scenario and identify applicable legal sections"""
        if not scenario_text or not isinstance(scenario_text, str):
            return {
                "crime_elements": [],
                "applicable_laws": [],
                "analysis": "Insufficient scenario details provided."
            }
            
        # Preprocess scenario text
        if self.nlp_engine:
            processed_text = self.nlp_engine.preprocess_text(scenario_text)
        else:
            processed_text = scenario_text
            
        # Extract crime elements from scenario
        crime_elements = self._extract_crime_elements(processed_text)
        
        # Find applicable laws
        applicable_laws = self._find_applicable_laws(crime_elements, processed_text)
        
        # Generate comprehensive analysis
        analysis = self._generate_analysis(crime_elements, applicable_laws, processed_text)
        
        return {
            "crime_elements": crime_elements,
            "applicable_laws": applicable_laws,
            "analysis": analysis
        }
        
    def _extract_crime_elements(self, text):
        """Extract crime elements from scenario text"""
        text_lower = text.lower()
        elements = []
        
        # Match crime patterns
        for crime_type, patterns in self.crime_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                elements.append(crime_type)
                
        # Extract entities if NLP engine is available
        if self.nlp_engine:
            entities = self.nlp_engine.extract_entities(text)
            
            # Add cyber crime entities
            if "CYBER_CRIME" in entities:
                for crime in entities["CYBER_CRIME"]:
                    if crime.lower() not in [e.lower() for e in elements]:
                        elements.append(crime)
        
        return elements
    
    def _find_applicable_laws(self, crime_elements, text):
        """Find all applicable laws based on crime elements and text"""
        applicable_laws = []
        
        # Search knowledge base for relevant laws
        if self.knowledge_base:
            # First try semantic search on the full text
            search_results = self.knowledge_base.search(text, top_n=5)
            
            # Filter to only include law results
            law_results = [result for result in search_results if result["type"] == "law"]
            
            # Add high and medium relevance results
            for result in law_results:
                if result["relevance"] in ["high", "medium"]:
                    law_info = result["content"]
                    applicable_laws.append({
                        "id": result["id"],
                        "title": law_info.get("title", "Unknown Law"),
                        "sections": law_info.get("legal_framework", {}).get("primary_sections", []),
                        "relevance": result["relevance"],
                        "description": law_info.get("description", ""),
                        "punishment": law_info.get("legal_framework", {}).get("punishment", "")
                    })
            
            # If no high/medium relevance results, search by crime elements
            if not applicable_laws:
                for element in crime_elements:
                    element_results = self.knowledge_base.search(element, top_n=3)
                    for result in element_results:
                        if result["type"] == "law":
                            law_info = result["content"]
                            applicable_laws.append({
                                "id": result["id"],
                                "title": law_info.get("title", "Unknown Law"),
                                "sections": law_info.get("legal_framework", {}).get("primary_sections", []),
                                "relevance": result["relevance"],
                                "description": law_info.get("description", ""),
                                "punishment": law_info.get("legal_framework", {}).get("punishment", "")
                            })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_laws = []
        for law in applicable_laws:
            if law["id"] not in seen:
                seen.add(law["id"])
                unique_laws.append(law)
                
        return unique_laws
    
    def _generate_analysis(self, crime_elements, applicable_laws, scenario_text):
        """Generate comprehensive analysis of the scenario"""
        if not crime_elements and not applicable_laws:
            return "No specific cyber crime elements were identified in this scenario. Please provide more details."
            
        analysis = "Based on the scenario provided, the following analysis applies:\n\n"
        
        # Add crime elements identified
        if crime_elements:
            analysis += "Criminal elements identified:\n"
            for i, element in enumerate(crime_elements, 1):
                # Convert snake_case to readable format
                readable_element = " ".join(element.split("_")).capitalize()
                analysis += f"{i}. {readable_element}\n"
            analysis += "\n"
            
        # Add applicable laws with sections
        if applicable_laws:
            analysis += "Applicable legal provisions:\n"
            for i, law in enumerate(applicable_laws, 1):
                analysis += f"{i}. {law['title']}\n"
                
                # Add sections
                if law["sections"]:
                    analysis += f"   Relevant sections: {', '.join(law['sections'])}\n"
                    
                # Add brief description
                if law["description"]:
                    analysis += f"   {law['description']}\n"
                    
                # Add punishment
                if law["punishment"]:
                    analysis += f"   Potential penalties: {law['punishment']}\n"
                    
                analysis += "\n"
        else:
            analysis += "No specific legal provisions were identified. Consider consulting a legal professional for more detailed analysis.\n"
            
        # Add disclaimer
        analysis += "\nDisclaimer: This analysis is provided for informational purposes only and should not be construed as legal advice. Please consult with a qualified legal professional for advice specific to your situation."
        
        return analysis


# Modify the CyberLawAssistant class to incorporate the ScenarioAnalyzer
class CyberLawAssistant:
    """Main application class integrating all components"""
    
    def __init__(self, use_gpu=False):
        self.logger = logging.getLogger("CyberLawAssistant")
        
        # Setup paths
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(self.data_path, exist_ok=True)
        
        # Initialize components
        self.nlp_engine = self._init_nlp_engine(use_gpu)
        self.knowledge_base = self._init_knowledge_base()
        self.conversation_manager = ConversationManager(max_history=20)
        self.response_generator = ResponseGenerator(self.knowledge_base, self.nlp_engine)
        self.scenario_analyzer = ScenarioAnalyzer(self.knowledge_base, self.nlp_engine)
        
        self.logger.info("Cyber Law Assistant initialized")
    
    def _init_nlp_engine(self, use_gpu):
        """Initialize NLP engine with error handling"""
        try:
            nlp_engine = NLPEngine(use_gpu=use_gpu)
            return nlp_engine
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP engine: {e}")
            return None
    
    def _init_knowledge_base(self):
        """Initialize knowledge base"""
        try:
            kb_path = os.path.join(self.data_path, "knowledge_base.json")
            knowledge_base = SemanticKnowledgeBase(kb_path, self.nlp_engine)
            return knowledge_base
        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge base: {e}")
            return None
    
    def process_input(self, user_input):
        """Process user input and generate response"""
        if not user_input or not isinstance(user_input, str):
            return "I couldn't understand that input. Please try again with a clear question about cyber law."
            
        try:
            # Check if this appears to be a crime scenario
            if self._is_crime_scenario(user_input):
                # Use scenario analyzer
                analysis = self.scenario_analyzer.analyze_scenario(user_input)
                response = analysis["analysis"]
            else:
                # Use regular response generator
                response = self.response_generator.generate_response(user_input, self.conversation_manager)
            
            # Add to conversation history
            self.conversation_manager.add_interaction(user_input, response, self.nlp_engine)
            
            return response
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            return "I encountered an error while processing your request. Please try again with a different question."
    
    def _is_crime_scenario(self, text):
        """Determine if input text appears to be a crime scenario description"""
        # Check length - scenarios tend to be longer
        if len(text.split()) < 15:  # Arbitrary threshold
            return False
            
        # Check for narrative markers
        narrative_markers = ["yesterday", "last week", "someone", "person", "individual", 
                           "received", "accessed", "stole", "hacked", "happened", 
                           "incident", "case", "scenario", "situation", "occurred"]
        
        text_lower = text.lower()
        marker_count = sum(1 for marker in narrative_markers if marker in text_lower)
        
        # Check for scenario request markers
        scenario_requests = ["analyze this", "this scenario", "this case", "what law", 
                           "which section", "sections apply", "legal provisions",
                           "provisions apply", "legal analysis"]
        
        request_marker = any(marker in text_lower for marker in scenario_requests)
        
        # Use NLP if available for better detection
        nlp_detection = False
        if self.nlp_engine and hasattr(self.nlp_engine, "classify_intent"):
            intent, confidence = self.nlp_engine.classify_intent(
                text, ["ask_question", "describe_scenario", "request_help", "report_crime"]
            )
            nlp_detection = intent in ["describe_scenario", "report_crime"] and confidence > 0.6
            
        # Combine signals
        return (marker_count >= 2) or request_marker or nlp_detection
    
    def analyze_scenario(self, scenario_text):
        """Direct access to scenario analysis functionality"""
        return self.scenario_analyzer.analyze_scenario(scenario_text)
    
    def save_session(self, filepath=None):
        """Save the current session"""
        if not filepath:
            filepath = os.path.join(self.data_path, 
                                  f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                                  
        return self.conversation_manager.save_conversation(filepath)
    
    def load_session(self, filepath):
        """Load a previous session"""
        return self.conversation_manager.load_conversation(filepath)
    
    def update_knowledge_base(self, new_data):
        """Update the knowledge base with new data"""
        if self.knowledge_base:
            return self.knowledge_base.update_knowledge_base(new_data)
        return False


def main():
    """Main function to run the assistant"""
    print("Initializing Cyber Law Assistant...")
    
    # Check for GPU availability
    use_gpu = False
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU detected, using CPU")
    except ImportError:
        print("PyTorch not installed or GPU not available, using CPU")
    
    # Initialize assistant
    assistant = CyberLawAssistant(use_gpu=use_gpu)
    
    print("\nCyber Law Assistant is ready!")
    print("Type your questions about cyber laws, crimes, reporting procedures, etc.")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("\nAssistant: Thank you for using the Cyber Law Assistant. Stay cyber-safe!")
                break
                
            if user_input:
                response = assistant.process_input(user_input)
                print(f"\nAssistant: {response}")
        except KeyboardInterrupt:
            print("\n\nSession terminated by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
    
    # Save session on exit
    session_path = os.path.join(assistant.data_path, f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    if assistant.save_session(session_path):
        print(f"\nSession saved to {session_path}")
    

if __name__ == "__main__":
     main()