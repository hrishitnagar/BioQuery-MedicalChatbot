"""
BioQuery – app.py (merged)
==========================
Combines:
  - Existing RAG pipeline (Pinecone + Gemini + LangChain)
  - New auth system (signup / login / logout)
  - SQLite user storage with bcrypt-hashed passwords
  - Session-based authentication

Install new dependency:
    pip install flask-bcrypt
"""

from flask import (
    Flask, render_template, jsonify, request,
    redirect, url_for, session, flash, g
)
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
from src.prompt import *
from functools import wraps
import sqlite3
import os


# ── App & extensions ──────────────────────────────────────────────────────────
app = Flask(__name__)

# Change this to a long random string in production!
# Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
app.secret_key = os.environ.get('SECRET_KEY', 'change-this-to-a-random-secret-in-production')

bcrypt = Bcrypt(app)

# ── Environment / API keys ────────────────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY   = os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"]   = GOOGLE_API_KEY


# ── RAG pipeline (unchanged from your original) ───────────────────────────────
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
docsearch  = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain             = create_retrieval_chain(retriever, question_answer_chain)


# ── SQLite database ───────────────────────────────────────────────────────────
DATABASE = 'bioquery.db'


def get_db():
    """Open a DB connection — one per request, reused within the same request."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row          # Access columns by name
    return db


@app.teardown_appcontext
def close_connection(exception):
    """Automatically close DB connection at the end of every request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def init_db():
    """Create the users table on first run (safe to call every startup)."""
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER  PRIMARY KEY AUTOINCREMENT,
                username    TEXT     NOT NULL UNIQUE,
                email       TEXT     NOT NULL UNIQUE,
                password    TEXT     NOT NULL,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.commit()


# ── Auth helper decorator ─────────────────────────────────────────────────────
def login_required(f):
    """Add @login_required above any route to restrict it to logged-in users."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access that page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route("/chat")
# @login_required   # <- Uncomment this line to require login before chatting
def index():
    return render_template('chat.html')


# ── Signup ────────────────────────────────────────────────────────────────────
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username         = request.form.get('username', '').strip()
        email            = request.form.get('email', '').strip().lower()
        password         = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        errors = []
        if not username or len(username) < 3:
            errors.append('Username must be at least 3 characters.')
        if not email or '@' not in email:
            errors.append('Please enter a valid email address.')
        if not password or len(password) < 8:
            errors.append('Password must be at least 8 characters.')
        if password != confirm_password:
            errors.append('Passwords do not match.')

        if errors:
            for err in errors:
                flash(err, 'error')
            return render_template('signup.html')

        db = get_db()

        existing = db.execute(
            'SELECT id FROM users WHERE username = ? OR email = ?',
            (username, email)
        ).fetchone()

        if existing:
            flash('Username or email is already registered. Please log in.', 'error')
            return render_template('signup.html')

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        db.execute(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            (username, email, hashed_pw)
        )
        db.commit()

        new_user = db.execute(
            'SELECT id FROM users WHERE username = ?', (username,)
        ).fetchone()

        session['user_id']  = new_user['id']
        session['username'] = username

        flash(f'Welcome to BioQuery, {username}! Your account has been created.', 'success')
        return redirect(url_for('index'))

    return render_template('signup.html')


# ── Login ─────────────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return render_template('login.html')

        db   = get_db()
        user = db.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()

        if user and bcrypt.check_password_hash(user['password'], password):
            session['user_id']  = user['id']
            session['username'] = user['username']
            flash(f'Welcome back, {user["username"]}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password. Please try again.', 'error')
            return render_template('login.html')

    return render_template('login.html')


# ── Logout ────────────────────────────────────────────────────────────────────
@app.route('/logout')
def logout():
    username = session.get('username', '')
    session.clear()
    flash(f'You have been signed out, {username}.', 'success')
    return redirect(url_for('landing'))


# ── Chat endpoint (unchanged from your original) ──────────────────────────────
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User Query: {msg}", flush=True)
    try:
        response = rag_chain.invoke({"input": msg})
        ans = response["answer"]
        print(f"Bot Response: {ans}", flush=True)
        return str(ans)
    except Exception as e:
        print(f"Error in RAG invocation: {str(e)}", flush=True)
        return f"Error: {str(e)}", 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()       # Creates bioquery.db + users table on first run
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)
