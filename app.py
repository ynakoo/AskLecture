import os
import streamlit as st
from groq import Groq
from src.embedding import TranscriptEmbedder
from src.retrieval import get_top_k

# Set page config
st.set_page_config(page_title="Semantic Video Search", page_icon="🎥", layout="wide")

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
st.title("🎥 Semantic Video Transcript Search")
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

# Handle API Key
env_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

# Sidebar for App Info & Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Allow user to input key if not in environment (don't pre-fill it)
    user_key = st.text_input("Override Groq API Key:", type="password", help="Leave blank to use the key from Streamlit Secrets/Environment.")
    
    # Final API key logic
    final_api_key = user_key if user_key else env_key
    
    if final_api_key:
        os.environ["GROQ_API_KEY"] = final_api_key
        st.session_state.api_key_valid = True
        if user_key:
            st.success("Using manual API key override.")
        else:
            st.info("Using API key from Secrets/Environment.")
    else:
        st.session_state.api_key_valid = False
        st.warning("Please enter your Groq API key to enable AI answers.")
        
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
                    
with tab_chat:
    if not st.session_state.stored_data:
        st.info("👈 Please upload and process a transcript first on the previous tab.")
    else:
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "context" in msg:
                    with st.expander("View Retrieved Context"):
                        for i, res in enumerate(msg["context"], 1):
                            st.write(f"**Chunk {i} (Score: {res['score']:.4f})**\n> {res['text']}")
                            
        # Chat Input
        if query := st.chat_input("Ask a question about the video transcript..."):
            # Append User Question
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
                
            # Retrieve Context
            with st.chat_message("assistant"):
                with st.spinner("Retrieving semantic matches..."):
                    query_emb = st.session_state.embedder.embed_query(query)
                    top_results = get_top_k(query_emb, st.session_state.stored_data, top_k=1)
                    
                if not top_results:
                    st.warning("No relevant context found in the transcript.")
                    context_texts = []
                else:
                    context_texts = [res["text"] for res in top_results]
                    
                combined_context = "\n\n".join(context_texts)
                
                # Check for Groq API
                if not st.session_state.api_key_valid or not combined_context:
                    if combined_context:
                        st.warning("API key not set. Below is the retrieved context only:")
                        for res in top_results:
                            st.info(res['text'])
                else:
                    with st.spinner("Generating answer..."):
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
                                    {"role": "user", "content": prompt}
                                ],
                                model="openai/gpt-oss-120b",
                            )
                            answer = chat_completion.choices[0].message.content
                            st.markdown(answer)
                            
                            # Save to chat history along with context UI
                            st.session_state.messages.append({"role": "assistant", "content": answer, "context": top_results})
                            
                            with st.expander("View Retrieved Context"):
                                for i, res in enumerate(top_results, 1):
                                    st.write(f"**Chunk {i} (Score: {res['score']:.4f})**\n> {res['text']}")
                                    
                        except Exception as e:
                            st.error(f"Error communicating with Groq API: {e}")
