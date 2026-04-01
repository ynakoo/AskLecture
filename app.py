import os
import streamlit as st
from groq import Groq
from src.embedding import TranscriptEmbedder
from src.retrieval import get_top_k

st.set_page_config(page_title="Semantic Video Search", page_icon="🎥", layout="wide")

st.markdown("""<style>...</style>""", unsafe_allow_html=True)

st.title("🎥 Semantic Video Transcript Search")
st.markdown("**Find answers directly from video transcripts...**")

# Session State Initialization
if "embedder" not in st.session_state:
    st.session_state.embedder = TranscriptEmbedder()
if "stored_data" not in st.session_state:
    st.session_state.stored_data = None
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set API Key automatically
os.environ["GROQ_API_KEY"] = "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"
st.session_state.api_key_valid = True

# Sidebar for App Info
with st.sidebar:
    st.header("⚙️ Configuration")

    st.divider()
    st.markdown("""
    ### About
    This app reads a plain text transcript, breaks it into semantic chunks, and searches exactly for what you're asking using the `sentence-transformers` library in-memory.
    """)

# Main UI
tab_upload, tab_chat = st.tabs(["📝 1. Provide Transcript", "💬 2. Ask Questions"])

with tab_upload:
    transcript = st.text_area("Paste your video transcript below:", height=250, placeholder="Artificial intelligence is...")
    
    if st.button("Process & Embed Transcript", use_container_width=True, type="primary"):
        if not transcript.strip():
            st.error("Please paste a valid transcript before processing.")
        else:
            with st.spinner("Chunking and generating embeddings locally... (this might take a moment)"):
                chunks = st.session_state.embedder.chunk_text(transcript.strip(), sentences_per_chunk=3)
                if not chunks:
                    st.error("Could not extract meaningful sentences from the transcript.")
                else:
                    st.session_state.stored_data = st.session_state.embedder.embed_chunks(chunks)
                    st.success(f"✅ Successfully processed {len(chunks)} contextual chunks into memory! You can now ask questions in the next tab.")