import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file automatically

import streamlit as st
import requests
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Set page config
st.set_page_config(page_title="AskLecture — Semantic Video Search", page_icon="🎥", layout="wide")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stTextArea textarea {
        background-color: #f7f9fc;
        color: #0f172a;
        caret-color: #0f172a;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        font-size: 0.95rem;
    }

    .stTextInput input {
        background-color: #f7f9fc;
        color: #0f172a;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        font-size: 0.95rem;
    }

    .stChatInput {
        border-radius: 20px;
    }

    h1 {
        background: linear-gradient(135deg, #1e3a8a, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }

    div[data-testid="stExpander"] {
        background-color: #f1f5f9;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }

    .success-box {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border: 1px solid #6ee7b7;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
    }

    .transcript-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        max-height: 400px;
        overflow-y: auto;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .badge-connected {
        background-color: #d1fae5;
        color: #065f46;
    }

    .badge-disconnected {
        background-color: #fee2e2;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "transcript_ready" not in st.session_state:
    st.session_state.transcript_ready = False
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = None

# ---------------------------------------------------------------------------
# API Key Setup
# ---------------------------------------------------------------------------
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    st.session_state.api_key_valid = True
except (KeyError, FileNotFoundError):
    if os.environ.get("GROQ_API_KEY"):
        st.session_state.api_key_valid = True
    else:
        st.session_state.api_key_valid = False


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def check_backend_health() -> bool:
    """Check if the FastAPI backend is reachable."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def send_process_text(text: str) -> dict:
    """Send transcript text to the backend for processing."""
    resp = requests.post(
        f"{BACKEND_URL}/process-text",
        json={"text": text},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def send_process_video(video_url: str) -> dict:
    """Send video URL to the backend for transcription + processing."""
    resp = requests.post(
        f"{BACKEND_URL}/process-video",
        json={"video_url": video_url},
        timeout=600,  # Transcription can take a while
    )
    resp.raise_for_status()
    return resp.json()


def send_retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve top-k relevant chunks from the backend."""
    resp = requests.post(
        f"{BACKEND_URL}/retrieve",
        json={"query": query, "top_k": top_k},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎥 AskLecture")
st.markdown("**Find answers from video transcripts using Groq LLM & local embeddings. Paste text or provide a video URL.**")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Backend status
    backend_ok = check_backend_health()
    if backend_ok:
        st.markdown('<span class="status-badge badge-connected">● Backend Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-disconnected">● Backend Offline</span>', unsafe_allow_html=True)
        st.error("FastAPI backend is not reachable. Start it with:\n```\nuvicorn backend.main:app --reload --port 8000\n```")

    if not st.session_state.api_key_valid:
        st.warning("⚠️ GROQ_API_KEY not found. Set it in `.env` or Streamlit Secrets.")

    st.divider()

    st.markdown("""
    ### About
    **AskLecture** reads video transcripts — pasted or auto-generated from video URLs — 
    breaks them into semantic chunks, and answers your questions using retrieval-augmented generation.

    **Architecture:**
    - 🖥️ **Frontend**: Streamlit  
    - ⚡ **Backend**: FastAPI  
    - 🎤 **Transcription**: Whisper (local)  
    - 🧠 **LLM**: Groq  
    """)

    st.divider()

    # Clear data button
    if st.button("🗑️ Clear All Data", use_container_width=True):
        try:
            requests.post(f"{BACKEND_URL}/clear", timeout=5)
            st.session_state.messages = []
            st.session_state.transcript_ready = False
            st.session_state.last_transcript = None
            st.success("Data cleared.")
            st.rerun()
        except Exception:
            st.error("Could not clear backend data.")

# ---------------------------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------------------------
