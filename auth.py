from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask import redirect, url_for, flash, request, session
import sqlite3
import os
import secrets
from datetime import datetime, timedelta

# Initialize login manager for Flask-Login
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

# Database initialization function
def init_db(app):
    db_path = os.path.join(app.instance_path, 'users.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            sender TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            rating INTEGER,
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (message_id) REFERENCES messages (id)
        )
        ''')
        
        conn.commit()
    
    # Add admin user if it doesn't exist
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = 'admin'")
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                ('admin', 'admin@example.com', generate_password_hash('admin'))
            )
            conn.commit()

@login_manager.user_loader
def load_user(user_id):
    db_path = os.path.join(current_app.instance_path, 'users.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email FROM users WHERE id = ?", (user_id,))
    user_data = cursor.fetchone()
    conn.close()
    
    if user_data:
        return User(id=user_data[0], username=user_data[1], email=user_data[2])
    return None

def register_user(username, email, password):
    db_path = os.path.join(current_app.instance_path, 'users.db')
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        if cursor.fetchone():
            conn.close()
            return False, "Username or email already exists"
        
        # Hash password and insert new user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hashed_password)
        )
        conn.commit()
        conn.close()
        return True, "Registration successful"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def authenticate_user(username_or_email, password):
    db_path = os.path.join(current_app.instance_path, 'users.db')
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if input is email or username
        if '@' in username_or_email:
            cursor.execute("SELECT id, username, email, password FROM users WHERE email = ?", (username_or_email,))
        else:
            cursor.execute("SELECT id, username, email, password FROM users WHERE username = ?", (username_or_email,))
        
        user_data = cursor.fetchone()
        conn.close()
        
        if user_data and check_password_hash(user_data[3], password):
            return User(id=user_data[0], username=user_data[1], email=user_data[2])
        return None
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        return None

def save_conversation(user_id, title, messages):
    """Save a conversation with its messages"""
    db_path = os.path.join(current_app.instance_path, 'users.db')
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create conversation
        cursor.execute(
            "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
            (user_id, title)
        )
        conversation_id = cursor.lastrowid
        
        # Save messages
        for message in messages:
            cursor.execute(
                "INSERT INTO messages (conversation_id, sender, content) VALUES (?, ?, ?)",
                (conversation_id, message['sender'], message['content'])
            )
        
        conn.commit()
        conn.close()
        return True, conversation_id
    except Exception as e:
        return False, f"Failed to save conversation: {str(e)}"

def get_user_conversations(user_id):
    """Get all conversations for a user"""
    db_path = os.path.join(current_app.instance_path, 'users.db')
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.id, c.title, c.created_at, 
                   (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count
            FROM conversations c
            WHERE c.user_id = ?
            ORDER BY c.created_at DESC
        """, (user_id,))
        
        conversations = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return conversations
    except Exception as e:
        print(f"Error getting conversations: {str(e)}")
        return []

def get_conversation_messages(conversation_id, user_id):
    """Get all messages for a conversation"""
    db_path = os.path.join(current_app.instance_path, 'users.db')
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # First verify that the conversation belongs to the user
        cursor.execute("SELECT id FROM conversations WHERE id = ? AND user_id = ?", 
                      (conversation_id, user_id))
        if not cursor.fetchone():
            conn.close()
            return None  # User doesn't own this conversation
        
        cursor.execute("""
            SELECT id, sender, content, timestamp
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """, (conversation_id,))
        
        messages = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return messages
    except Exception as e:
        print(f"Error getting messages: {str(e)}")
        return None

def save_feedback(message_id, rating, comment=None):
    """Save feedback for a bot message"""
    db_path = os.path.join(current_app.instance_path, 'users.db')
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO feedback (message_id, rating, comment) VALUES (?, ?, ?)",
            (message_id, rating, comment)
        )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        return False

# Import current_app for the user_loader function
from flask import current_app