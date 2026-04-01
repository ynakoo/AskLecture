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