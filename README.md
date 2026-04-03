# 🎥 AskLecture — Semantic Video Transcript Search

> Ask questions about any lecture or video transcript, upload audio for transcription, and get precise, AI-powered answers instantly.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://asklecture-s2npayyxbeks7nr5fcdfqr.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)

---

## 📌 What is AskLecture?

**AskLecture** is a Retrieval-Augmented Generation (RAG) application that lets you paste any video/lecture transcript and ask natural language questions about it. It uses local sentence embeddings for semantic search and the Groq API for LLM-powered answer generation.

---

## ✨ Features

- 📝 **Paste any transcript or upload audio** — paste text directly or upload audio files for automatic transcription via AssemblyAI
- 🔍 **Semantic search** — finds the most relevant chunks using cosine similarity
- 🤖 **AI-powered answers** — generates concise answers using Groq's LLM (GPT-OSS-120B)
- 💬 **Chat interface** — conversational UI with full chat history
- 📎 **Context transparency** — view the exact retrieved chunks used for each answer
- ⚡ **Local embeddings** — uses `all-MiniLM-L6-v2` (no external API needed for embeddings)
- 🖥️ **CLI mode** — also available as a command-line tool

---

## 🏗️ Architecture

```
AskLecture/
├── app.py                 # Streamlit web application
├── cli.py                 # Command-line interface
├── requirements.txt       # Python dependencies
├── src/
│   ├── embedding.py       # Text chunking & embedding (SentenceTransformers)
│   └── retrieval.py       # Cosine similarity search (top-k retrieval)
└── notebooks/
    └── rag_app.ipynb      # Jupyter notebook version
```

### How It Works

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐
│  Paste Text /│ ──▶ │  Chunk Text  │ ──▶ │  Generate     │ ──▶ │  Store in   │
│  Upload Audio│     │  (3 sent.)   │     │  Embeddings   │     │  Memory     │
└─────────────┘     └──────────────┘     └───────────────┘     └─────────────┘
                                                                      │
┌─────────────┐     ┌──────────────┐     ┌───────────────┐           │
│  Display     │ ◀── │  Groq LLM    │ ◀── │  Retrieve     │ ◀─────────┘
│  Answer      │     │  Generation  │     │  Top-K Chunks │
└─────────────┘     └──────────────┘     └───────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A Groq API key ([get one here](https://console.groq.com/keys))

### Installation

```bash
# Clone the repository
git clone https://github.com/ynakoo/AskLecture.git
cd AskLecture

# Install dependencies
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Run the CLI Version

```bash
python cli.py
```

Paste your transcript, type `DONE`, and start asking questions.

---

## 🌐 Live Demo

The app is deployed on Streamlit Community Cloud:

🔗 **[asklecture-s2npayyxbeks7nr5fcdfqr.streamlit.app](https://asklecture-s2npayyxbeks7nr5fcdfqr.streamlit.app/)**

---

## 🛠️ Tech Stack

| Component       | Technology                          |
| --------------- | ----------------------------------- |
| **Frontend**    | Streamlit                           |
| **Transcription**| AssemblyAI API                      |
| **Embeddings**  | SentenceTransformers (all-MiniLM-L6-v2) |
| **Similarity**  | scikit-learn (Cosine Similarity)    |
| **LLM**        | Groq API (GPT-OSS-120B)            |
| **Language**    | Python 3.10+                        |

---

## 📖 Usage

1. **Open the app** in your browser
2. **Go to the "Provide Transcript" tab** and paste your transcript or upload an audio file for automatic transcription
3. **Click "Process & Embed Transcript"** to chunk and embed the text
4. **Switch to the "Ask Questions" tab** and start chatting
5. **Expand "View Retrieved Context"** to see which chunks were used

---

