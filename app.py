import os
import streamlit as st
from groq import Groq
from src.embedding import TranscriptEmbedder
from src.retrieval import get_top_k

st.set_page_config(page_title="Semantic Video Search", page_icon="🎥", layout="wide")

st.markdown("""<style>...</style>""", unsafe_allow_html=True)

st.title("🎥 Semantic Video Transcript Search")
st.markdown("**Find answers directly from video transcripts...**")