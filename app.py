import os
import streamlit as st
import assemblyai as aai
from dotenv import load_dotenv
from groq import Groq
from src.embedding import TranscriptEmbedder
from src.retrieval import get_top_k

# Load environment variables from .env
load_dotenv()

# AssemblyAI Config
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# Set page config
st.set_page_config(page_title="🎥 AskLecture", page_icon="🎥", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .stTextArea textarea {
        background-color: #f7f9fc;
        color: #000000;
        caret-color: #000000;
        border-radius: 8px;
    }
    .stChatInput {
        border-radius: 20px;
    }
    h1 {
        color: #1e3a8a;
    }
</style>
""", unsafe_allow_html=True)

# Application Heading
st.title("🎥 AskLecture")
st.markdown("**Find answers directly from video transcripts using Groq LLaMA 3 & local embeddings.**")

# Session State Initialization
if "embedder" not in st.session_state:
    st.session_state.embedder = TranscriptEmbedder()
if "stored_data" not in st.session_state:
    st.session_state.stored_data = None
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "full_transcript" not in st.session_state:
    st.session_state.full_transcript = ""
if "processing" not in st.session_state:
    st.session_state.processing = False

def start_processing():
    st.session_state.processing = True

def transcribe_audio(file):
    """Uses AssemblyAI SDK to transcribe uploaded audio file."""
    if not aai.settings.api_key:
        st.error("❌ AssemblyAI API Key not found. Please set `ASSEMBLYAI_API_KEY` environment variable.")
        return None
    
    try:
        # Configure transcription to use universal models (now mandatory)
        config = aai.TranscriptionConfig(speech_models=["universal-3-pro", "universal-2"])
        transcriber = aai.Transcriber()
        # The SDK can take a file-like object directly
        transcript = transcriber.transcribe(file, config=config)
        
        if transcript.status == aai.TranscriptStatus.error:
            st.error(f"❌ Transcription Error: {transcript.error}")
            return None
            
        return transcript.text
    except Exception as e:
        st.error(f"❌ An error occurred during transcription: {str(e)}")
        return None

# Handle API Key Safely (Silent)
detected_key = os.environ.get("GROQ_API_KEY")
if not detected_key:
    try:
        detected_key = st.secrets.get("GROQ_API_KEY")
    except (FileNotFoundError, KeyError, RuntimeError):
        detected_key = None

st.session_state.api_key_valid = True if detected_key else False

# Sidebar for App Info & Configuration
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
    st.subheader("📂 Step 1: Provide Source")
    
    # Audio Uploader (Top of section as requested)
    # Audio Uploader
    audio_file = st.file_uploader(
        "Option A: Upload Audio (mp3, wav, m4a)", 
        type=["mp3", "wav", "m4a"],
        key=f"audio_uploader_{st.session_state.uploader_key}"
    )
    
    if audio_file:
        if st.button("🎙️ Transcribe Audio", use_container_width=True, disabled=st.session_state.processing, on_click=start_processing):
            with st.spinner("🎙️ Transcribing audio using AssemblyAI..."):
                result = transcribe_audio(audio_file)
                if result:
                    st.session_state.full_transcript = result
                    st.success("✅ Transcription complete! The text has been added below for your review.")
                st.session_state.processing = False
                st.rerun()
    
    st.markdown("--- OR ---")
    
    # Text Input (Option B & Transcription target)
    transcript_text = st.text_area(
        "Option B: Paste or Review Transcript Text", 
        height=200, 
        placeholder="Paste your video transcript here or transcribe audio above...",
        value=st.session_state.full_transcript
    )
    
    # Keep session state in sync with manual edits
    if transcript_text != st.session_state.full_transcript:
        st.session_state.full_transcript = transcript_text
    
    if st.button("🚀 Process & Embed", use_container_width=True, type="primary", disabled=st.session_state.processing, on_click=start_processing):
        if not st.session_state.full_transcript.strip():
            st.error("Please provide a transcript (either by pasting text or transcribing audio) before processing.")
            st.session_state.processing = False
            st.rerun()
        else:
            with st.spinner("🧠 Chunking and generating embeddings locally... (this might take a moment)"):
                chunks = st.session_state.embedder.chunk_text(st.session_state.full_transcript, sentences_per_chunk=3)
                if not chunks:
                    st.error("Could not extract meaningful sentences from the transcript.")
                else:
                    st.session_state.stored_data = st.session_state.embedder.embed_chunks(chunks)
                    st.success(f"✅ Successfully processed {len(chunks)} contextual chunks into memory! You can now ask questions in the next tab.")
                    
                    # Reset uploader to allow new file if desired
                    if audio_file:
                        st.session_state.uploader_key += 1
                
                st.session_state.processing = False
                st.rerun()
                    
with tab_chat:
    if not st.session_state.stored_data:
        st.info("👈 Please upload and process a transcript first on the previous tab.")
    else:
        # Chat Input at the Top
        with st.form("chat_form", clear_on_submit=True):
            query = st.text_input("💬 Ask a question about the video transcript...", placeholder="Type your question and press Enter...", disabled=st.session_state.processing)
            submit_button = st.form_submit_button("Ask", disabled=st.session_state.processing, on_click=start_processing)
            
        if submit_button and query.strip():
            user_query = query.strip()
            # Append User Question
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Retrieve Context
            with st.spinner("🤔 Searching for relevant context..."):
                query_emb = st.session_state.embedder.embed_query(user_query)
                top_results = get_top_k(query_emb, st.session_state.stored_data, top_k=1)
            
            if not top_results:
                st.session_state.messages.append({"role": "assistant", "content": "I couldn't find any relevant context in the transcript to answer that."})
            else:
                context_texts = [res["text"] for res in top_results]
                combined_context = "\n\n".join(context_texts)
                
                # Check for Groq API
                if not st.session_state.api_key_valid:
                    st.session_state.messages.append({"role": "assistant", "content": "Groq API key not set. Here is the relevant context found:\n\n" + combined_context})
                else:
                    try:
                        with st.spinner("🧠 Generating answer..."):
                            client = Groq(api_key=detected_key)
                            prompt = f"""You are a helpful assistant. Answer the question ONLY using the provided context. If the answer is not in the context, say 'I don't know'.

Context:
{combined_context}

Question:
{user_query}

Answer clearly and concisely."""

                            chat_completion = client.chat.completions.create(
                                messages=[
                                    {"role": "system", "content": "You are a precise and helpful assistant."},
                                    {"role": "user", "content": prompt}
                                ],
                                model="openai/gpt-oss-120b",
                            )
                            answer = chat_completion.choices[0].message.content
                            st.session_state.messages.append({"role": "assistant", "content": answer, "context": top_results})
                    except Exception as e:
                        error_msg = f"❌ Error communicating with API: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            # Reset processing state
            st.session_state.processing = False
            # Use rerun to clear the input box and show new messages
            st.rerun()

        st.divider()

        # Display chat history
        if not st.session_state.messages:
            st.write("Start the conversation by asking a question above!")
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "context" in msg:
                        with st.expander("View Retrieved Context"):
                            for i, res in enumerate(msg["context"], 1):
                                st.write(f"**Chunk {i} (Score: {res['score']:.4f})**\n> {res['text']}")
