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
if "processing" not in st.session_state:
    st.session_state.processing = False

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
    if st.button("🗑️ Clear All Data", use_container_width=True, disabled=st.session_state.processing):
        st.session_state.processing = True
        try:
            requests.post(f"{BACKEND_URL}/clear", timeout=5)
            st.session_state.messages = []
            st.session_state.transcript_ready = False
            st.session_state.last_transcript = None
            st.session_state.processing = False
            st.success("Data cleared.")
            st.rerun()
        except Exception:
            st.session_state.processing = False
            st.error("Could not clear backend data.")

# ---------------------------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------------------------
tab_text, tab_video, tab_chat = st.tabs(["📝 Paste Transcript", "🎬 Video URL", "💬 Ask Questions"])

# ---- Tab 1: Paste Transcript ----
with tab_text:
    st.subheader("Paste a Transcript")
    st.caption("Paste your lecture or video transcript below, then click Process.")

    transcript = st.text_area(
        "Transcript text:",
        height=250,
        placeholder="Artificial intelligence is a branch of computer science that aims to create intelligent machines...",
        label_visibility="collapsed",
    )

    if st.button("⚡ Process & Embed Transcript", use_container_width=True, type="primary", key="btn_process_text", disabled=st.session_state.processing):
        st.session_state.processing = True
        if not transcript.strip():
            st.session_state.processing = False
            st.error("Please paste a valid transcript before processing.")
        elif not backend_ok:
            st.session_state.processing = False
            st.error("Backend is offline. Please start the FastAPI server first.")
        else:
            with st.spinner("Sending transcript to backend for chunking & embedding..."):
                try:
                    result = send_process_text(transcript.strip())
                    num_chunks = result.get("num_chunks", 0)
                    st.session_state.transcript_ready = True
                    st.session_state.last_transcript = transcript.strip()
                    st.session_state.processing = False
                    st.markdown(
                        f'<div class="success-box">✅ Successfully processed <strong>{num_chunks}</strong> '
                        f'chunks into memory! Head to the <strong>💬 Ask Questions</strong> tab.</div>',
                        unsafe_allow_html=True,
                    )
                except requests.HTTPError as e:
                    st.session_state.processing = False
                    detail = ""
                    try:
                        detail = e.response.json().get("detail", "")
                    except Exception:
                        pass
                    st.error(f"Backend error: {detail or e}")
                except requests.ConnectionError:
                    st.session_state.processing = False
                    st.error("Cannot reach backend. Is the FastAPI server running?")

# ---- Tab 2: Video URL ----
with tab_video:
    st.subheader("Transcribe from Video URL")
    st.caption("Provide a YouTube or other video URL. The backend will extract audio and transcribe it using Whisper.")

    video_url = st.text_input(
        "Video URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )

    if st.button("🎬 Convert & Process Video", use_container_width=True, type="primary", key="btn_process_video", disabled=st.session_state.processing):
        st.session_state.processing = True
        if not video_url.strip():
            st.session_state.processing = False
            st.error("Please enter a valid video URL.")
        elif not backend_ok:
            st.session_state.processing = False
            st.error("Backend is offline. Please start the FastAPI server first.")
        else:
            with st.spinner("⏳ Downloading audio, transcribing with Whisper, and generating embeddings... This may take a few minutes."):
                try:
                    result = send_process_video(video_url.strip())
                    num_chunks = result.get("num_chunks", 0)
                    transcript_text = result.get("transcript", "")

                    st.session_state.transcript_ready = True
                    st.session_state.last_transcript = transcript_text
                    st.session_state.processing = False

                    st.markdown(
                        f'<div class="success-box">✅ Video transcribed and processed into <strong>{num_chunks}</strong> '
                        f'chunks! Head to the <strong>💬 Ask Questions</strong> tab.</div>',
                        unsafe_allow_html=True,
                    )

                    # Show transcript
                    if transcript_text:
                        st.markdown("#### 📄 Generated Transcript")
                        st.markdown(
                            f'<div class="transcript-box">{transcript_text}</div>',
                            unsafe_allow_html=True,
                        )

                        # Download button
                        st.download_button(
                            label="📥 Download Transcript",
                            data=transcript_text,
                            file_name="transcript.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )

                except requests.HTTPError as e:
                    st.session_state.processing = False
                    detail = ""
                    try:
                        detail = e.response.json().get("detail", "")
                    except Exception:
                        pass
                    st.error(f"❌ Backend error: {detail or e}")
                except requests.ConnectionError:
                    st.session_state.processing = False
                    st.error("Cannot reach backend. Is the FastAPI server running?")
                except requests.Timeout:
                    st.session_state.processing = False
                    st.error("⏰ Request timed out. The video might be too long. Try a shorter video.")

# ---- Tab 3: Ask Questions ----
with tab_chat:
    if not st.session_state.transcript_ready:
        st.info("👈 Please process a transcript or video first using one of the other tabs.")
    else:
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "context" in msg:
                    with st.expander("📎 View Retrieved Context"):
                        for i, res in enumerate(msg["context"], 1):
                            st.write(f"**Chunk {i} (Score: {res['score']:.4f})**\n> {res['text']}")

        # Chat input
        if query := st.chat_input("Ask a question about the transcript..."):
            # Show user message
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # Retrieve & answer
            with st.chat_message("assistant"):
                # Step 1: Retrieve relevant chunks from backend
                with st.spinner("🔍 Retrieving relevant context..."):
                    try:
                        top_results = send_retrieve(query, top_k=3)
                    except Exception as e:
                        st.error(f"Retrieval failed: {e}")
                        top_results = []

                if not top_results:
                    st.warning("No relevant context found in the transcript.")
                    context_texts = []
                else:
                    context_texts = [res["text"] for res in top_results]

                combined_context = "\n\n".join(context_texts)

                # Step 2: Query Groq LLM (stays in frontend)
                if not st.session_state.api_key_valid or not combined_context:
                    if combined_context:
                        st.warning("API key not set. Below is the retrieved context only:")
                        for res in top_results:
                            st.info(res["text"])
                else:
                    with st.spinner("🧠 Generating answer with Groq LLM..."):
                        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                        prompt = f"""You are a helpful assistant. Answer the question ONLY using the provided context. If the answer is not in the context, say 'I don't know'.

Context:
{combined_context}

Question:
{query}

Answer clearly and concisely."""

                        try:
                            chat_completion = client.chat.completions.create(
                                messages=[
                                    {"role": "system", "content": "You are a precise and helpful assistant."},
                                    {"role": "user", "content": prompt},
                                ],
                                model="openai/gpt-oss-120b",
                            )
                            answer = chat_completion.choices[0].message.content
                            st.markdown(answer)

                            # Save to history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "context": top_results,
                            })

                            with st.expander("📎 View Retrieved Context"):
                                for i, res in enumerate(top_results, 1):
                                    st.write(f"**Chunk {i} (Score: {res['score']:.4f})**\n> {res['text']}")

                        except Exception as e:
                            st.error(f"Error communicating with Groq API: {e}")