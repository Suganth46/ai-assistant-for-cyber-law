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
        print("Initialize NLP components")
        # Load sentence embedding model
        try:
            
            self.embedding_model = SentenceTransformer('all-mpnet-base-v2')  # More powerful model
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
        @Language.component("legal_term_tagger")
        def legal_term_tagger(doc):
            """Tag legal terms with higher precision"""
            legal_terms = {
            "hacking": {"POS": "NOUN", "TAG": "NN", "DEP": "dobj"},
            "accessed": {"POS": "VERB", "TAG": "VBD", "DEP": "ROOT"},
            "breach": {"POS": "NOUN", "TAG": "NN", "DEP": "dobj"},
            "violation": {"POS": "NOUN", "TAG": "NN", "DEP": "dobj"},
            "prosecution": {"POS": "NOUN", "TAG": "NN", "DEP": "pobj"},
            "defendant": {"POS": "NOUN", "TAG": "NN", "DEP": "nsubj"},
            "legislation": {"POS": "NOUN", "TAG": "NN", "DEP": "pobj"}
            }
    
            for token in doc:
                lower_text = token.text.lower()
                if lower_text in legal_terms:
                    token.tag_ = legal_terms[lower_text]["TAG"]
                    token.pos_ = legal_terms[lower_text]["POS"]
            
            return doc    
        # Add component to pipeline
        self.nlp.add_pipe("cyber_law_entities", after="ner")
        self.nlp.add_pipe("legal_term_tagger", after="cyber_law_entities")
    
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
        print("Analyze the sentiment and emotional content of text")
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
        """Enhanced preprocessing for better accuracy with robust error handling"""
        print("Enhanced preprocessing for better accuracy with robust error handling")
        try:
            # Input validation
            if not text or not isinstance(text, str):
                return ""
        
            # Basic cleanup
            text = text.strip()
        
            # Remove excessive whitespace (including newlines, tabs)
            text = re.sub(r'\s+', ' ', text)
        
            # Normalize punctuation for better sentence boundary detection
            # More comprehensive pattern that handles multiple punctuation
            text = re.sub(r'([.!?])\s*([.!?])+', r'\1', text)
        
            # Fix common punctuation issues
            text = re.sub(r'([.!?]),', r'\1', text)  # Remove commas after sentence endings
            text = re.sub(r'([.!?])([a-zA-Z])', r'\1 \2', text)  # Ensure space after punctuation
        
            # Standardize quotes
            text = re.sub(r'[""]', '"', text)
            text = re.sub(r'['']', "'", text)
        
            # Handle brackets and parentheses consistently
            text = re.sub(r'\s*\(\s*', ' (', text)  # Space before open parenthesis
            text = re.sub(r'\s*\)\s*', ') ', text)  # Space after close parenthesis
        
            # Expand common abbreviations relevant to legal text
            abbreviations = {
                r'\bu/?s\b': 'under section',
                r'\bsec\.?\b': 'section',
                r'\bIT Act\b': 'Information Technology Act',
                r'\bIPC\b': 'Indian Penal Code',
                r'\bNIIA\b': 'National Information Infrastructure Act',
                r'\bcyber\s*crime\b': 'cybercrime',
                r'\bcfaa\b': 'Computer Fraud and Abuse Act',
                r'\becpa\b': 'Electronic Communications Privacy Act',
                r'\bgdpr\b': 'General Data Protection Regulation',
                r'\bcoppa\b': 'Children\'s Online Privacy Protection Act',
                r'\bCISO\b': 'Chief Information Security Officer',
                r'\bdns\b': 'Domain Name System',
                r'\bpii\b': 'personally identifiable information'
            }
        
            for abbr, expanded in abbreviations.items():
                text = re.sub(abbr, expanded, text, flags=re.IGNORECASE)
        
            # Remove URLs if they might cause issues in further processing
            # text = re.sub(r'https?://\S+', '[URL]', text)
        
            # Normalize sentence spacing
            text = re.sub(r'([.!?])\s+', r'\1 ', text)

            # Remove duplicate spaces (again, after all other operations)
            text = re.sub(r'\s+', ' ', text)
        
            # Final trim
            text = text.strip()

            return text

        except Exception as e:
            # Log the error for debugging
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in text preprocessing: {str(e)}")

            # Safely return original text if preprocessing fails
            # This ensures the system can continue functioning
            return text.strip() if isinstance(text, str) else ""
    def analyze_sentences(self, text: str) -> List[Dict]:
        """Break down text into sentences with detailed analysis"""
        print("Break down text into sentences with detailed analysis")
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            sentences = []
        
            for sent in doc.sents:
                # Get key information from each sentence
                root_verb = None
                subject = None
            
                for token in sent:
                    if token.dep_ == "ROOT" and token.pos_ == "VERB":
                        root_verb = token.text
                    if token.dep_.startswith("nsubj") and token.pos_ in ["NOUN", "PROPN", "PRON"]:
                        subject = token.text
            
                sentences.append({
                    "text": sent.text,
                    "entities": [{"text": ent.text, "label": ent.label_} for ent in sent.ents],
                    "root_verb": root_verb,
                    "subject": subject,
                    "tokens": [{"text": token.text, "pos": token.pos_, "dep": token.dep_} for token in sent]
                })

            return sentences
        except Exception as e:
            self.logger.error(f"Error analyzing sentences: {e}")
            return []

class ConversationManager:
    """Enhanced conversation manager with semantic understanding"""
    print("Enhanced conversation manager with semantic understanding")
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
                # return self._create_enhanced_knowledge_base()
            
            # Load the file
            with open(filepath, 'r') as file:
                data = json.load(file)
                logger.info(f"Knowledge base loaded successfully from {filepath}")
                return data
                
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            # return self._create_enhanced_knowledge_base()
    

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
        """Enhanced semantic search with better similarity handling"""
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
            
                # Calculate cosine similarity
                sim_score = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    item_embedding.reshape(1, -1)
                )[0][0]
            
                # Apply length normalization - helps with short vs long text comparison
                query_words = len(query.split())
                bias = 1.0
                if query_words < 5:
                    bias = 0.8  # Short queries need adjustment
                elif query_words > 20:
                    bias = 1.2  # Longer queries are often more specific
                
                adjusted_score = sim_score * bias
            
                similarities.append((key, adjusted_score, item))
        
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
        
            # Return top N results with improved relevance criteria
            results = []
            for key, similarity, item in similarities[:top_n]:
                if similarity < 0.25:  # Lower threshold to catch more potential matches
                    continue
                
                # More granular relevance classification
                if similarity > 0.7:
                    relevance = "high"
                elif similarity > 0.5:
                    relevance = "medium"
                elif similarity > 0.3:
                    relevance = "low"
                else:
                    relevance = "marginal"
                
                if item["type"] == "law":
                    law_id = item["id"]
                    law_info = self.knowledge_base["laws"].get(law_id, {})
                    results.append({
                        "type": "law",
                        "id": law_id,
                        "content": law_info,
                        "similarity": similarity,
                        "relevance": relevance
                    })
                elif item["type"] == "intent":
                    intent_tag = item["id"]
                    intent_info = next((i for i in self.knowledge_base["intents"] if i.get("tag") == intent_tag), {})
                    results.append({
                        "type": "intent",
                        "id": intent_tag,
                        "content": intent_info,
                        "similarity": similarity,
                        "relevance": relevance
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
    print("Analyze cyber crime scenarios and map them to relevant legal sections")
    def __init__(self, knowledge_base, nlp_engine):
        self.knowledge_base = knowledge_base
        self.nlp_engine = nlp_engine
        self.crime_patterns = self._load_crime_patterns()
        
    def _load_crime_patterns(self):
        """Load patterns for recognizing common elements in cyber crime scenarios"""
        return {
            "unauthorized_access": [
                "hack", "broke into", "unauthorized access", "illegal access", "gain access", 
                "breach", "compromise", "infiltrate", "login without permission", "bypass security",
                "exploit vulnerability", "zero-day exploit", "privilege escalation", "credential stuffing"
            ],
            "data_theft": [
                "stole data", "took information", "downloaded files", "exfiltrated", "copied data",
                "data theft", "credential theft", "stole credentials", "identity theft", "data breach",
                "leaked data", "exposed data", "sensitive information", "personal data", "financial data",
                "trade secrets", "confidential information"
            ],
            "malware": [
                "malware", "virus", "ransomware", "trojan", "worm", "spyware", "keylogger",
                "infected", "malicious software", "malicious code", "crypto malware", "botnet",
                "rootkit", "backdoor", "adware", "fileless malware", "polymorphic malware"
            ],
            "denial_of_service": [
                "dos attack", "ddos", "denial of service", "crashed server", "flooded",
                "made unavailable", "disrupted service", "overwhelm server", "traffic flood",
                "botnet attack", "amplification attack", "reflection attack"
            ],
            "fraud": [
                "fraud", "deceived", "impersonated", "fake", "pretended to be", "scam", 
                "fraudulent", "misrepresented", "social engineering", "phishing", "spear phishing",
                "whaling", "vishing", "smishing", "business email compromise", "romance scam",
                "investment fraud", "crypto scam", "fake website", "spoofing"
            ],
            "harassment": [
                "harass", "stalk", "threaten", "blackmail", "extort", "intimidate",
                "bullying", "threatening messages", "defame", "cyberstalking", "doxing",
                "revenge porn", "online harassment", "hate speech", "trolling", "swatting"
            ],
            "sexual_offenses": [
                "obscene", "explicit content", "child exploitation", "revenge porn",
                "non-consensual", "intimate images", "sexually explicit", "child pornography",
                "sextortion", "grooming", "online sexual abuse", "sexual harassment"
            ],
            "intellectual_property": [
                "copyright", "piracy", "illegal download", "counterfeit", "plagiarize",
                "intellectual property", "trademark", "patent", "software piracy", "movie piracy",
                "music piracy", "ebook piracy", "trade secret theft", "industrial espionage"
            ],
            "cryptocurrency_crimes": [
                "crypto theft", "crypto fraud", "crypto scam", "crypto mining malware",
                "cryptojacking", "crypto wallet theft", "crypto exchange hack", "crypto phishing",
                "crypto ransomware", "crypto laundering", "crypto pump and dump"
            ],
            "supply_chain_attacks": [
                "supply chain attack", "third-party breach", "vendor compromise", "software supply chain",
                "dependency confusion", "typosquatting", "dependency hijacking", "package poisoning"
            ],
            "advanced_persistent_threat": [
                "apt", "advanced persistent threat", "nation state attack", "state-sponsored",
                "cyber espionage", "targeted attack", "long-term infiltration", "covert operation"
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
        """Process user input with enhanced accuracy"""
        if not user_input or not isinstance(user_input, str):
            return "I couldn't understand that input. Please try again with a clear question about cyber law."
             
        # First verify that nlp_engine exists
        if not hasattr(self, 'nlp_engine') or self.nlp_engine is None:
            self.logger.error("NLP Engine is not initialized")
            return "System error: NLP engine not available. Please contact support."
    
        try:
            # Preprocess the input for better analysis
            try:
                preprocessed_input = self.nlp_engine.preprocess_text(user_input)
                self.logger.debug(f"Preprocessed input: {preprocessed_input}")
            except Exception as e:
                self.logger.error(f"Error during preprocessing: {str(e)}")
                return "Error occurred during text preprocessing. Please try again with simpler text."

            # Rest of the code with more specific error handling...

            # For short queries, try this simplified path first
            if len(preprocessed_input.split()) <= 7:
                try:
                    response = self.response_generator.generate_response(user_input, self.conversation_manager)
                    # Add to conversation history
                    self.conversation_manager.add_interaction(user_input, response, self.nlp_engine)
                    return response
                except Exception as e:
                    self.logger.error(f"Error processing short input: {str(e)}")
                    return "Error processing your question. Please try rephrasing it."
                
            # For longer text, proceed with more complex processing
            # ...rest of the code with additional try/except blocks
        
        except Exception as e:
            self.logger.error(f"Unhandled error processing input: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())  # Log the full stack trace
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