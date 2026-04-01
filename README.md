# 🎥 AskLecture

**Find answers from video transcripts using Groq LLM & local embeddings.**

AskLecture is a semantic search application that lets you ask natural language questions about video content. Paste a transcript or provide a video URL — the system will transcribe, chunk, embed, and answer using retrieval-augmented generation (RAG).

---

## Architecture

```
Streamlit Frontend (UI)
       ↓ HTTP
FastAPI Backend (processing)
       ├── POST /process-text   → chunk + embed + store
       ├── POST /process-video  → yt-dlp → Whisper → chunk + embed + store
       ├── POST /retrieve       → semantic search over stored chunks
       ├── POST /clear          → reset stored data
       └── GET  /health         → health check
       ↓
Shared Modules:
       ├── src/embedding.py     → SentenceTransformer embeddings
       └── src/retrieval.py     → cosine similarity retrieval
```

## Features

- **Paste Transcript** — paste any text and process it for Q&A
- **Video URL** — provide a YouTube (or other) video URL:
  - Audio extracted via `yt-dlp`
  - Transcribed locally using OpenAI Whisper (`base` model)
  - Automatically processed into the same RAG pipeline
  - Download the generated transcript
- **Ask Questions** — chat interface with context-aware answers via Groq LLM
- **Clean Separation** — Streamlit handles UI, FastAPI handles all processing

---

## Project Structure

```
AskLecture/
├── app.py                          # Streamlit frontend (UI only)
├── cli.py                          # CLI client
├── backend/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application
│   ├── audio.py                    # yt-dlp audio extraction
│   └── whisper_transcribe.py       # Whisper transcription
├── src/
│   ├── embedding.py                # SentenceTransformer embeddings
│   └── retrieval.py                # Cosine similarity retrieval
├── requirements.txt
├── .env.example
└── README.md
```

---

## Prerequisites

- **Python 3.10+**
- **ffmpeg** — required for audio extraction and Whisper

### Install ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
```

---

## Setup

### 1. Clone & Install Dependencies

```bash
git clone <your-repo-url>
cd AskLecture
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Start the FastAPI Backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start the Streamlit Frontend (in a separate terminal)

```bash
streamlit run app.py
```

### 5. Open the App

Visit `http://localhost:8501` in your browser.

---

## Usage

### Option 1: Paste a Transcript
1. Go to the **📝 Paste Transcript** tab
2. Paste your lecture/video transcript
3. Click **Process & Embed Transcript**
4. Switch to **💬 Ask Questions** and chat

### Option 2: Video URL
1. Go to the **🎬 Video URL** tab
2. Enter a YouTube or other video URL
3. Click **Convert & Process Video**
4. Wait for transcription (may take a few minutes)
5. View and download the generated transcript
6. Switch to **💬 Ask Questions** and chat

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check + status |
| `/process-text` | POST | Process pasted transcript text |
| `/process-video` | POST | Download, transcribe, and process a video |
| `/retrieve` | POST | Retrieve top-k relevant chunks for a query |
| `/clear` | POST | Clear all stored embeddings |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Your Groq API key |
| `BACKEND_URL` | `http://localhost:8000` | FastAPI backend URL |
| `WHISPER_MODEL` | `base` | Whisper model size: tiny, base, small, medium, large |

---

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI + Uvicorn
- **Transcription**: OpenAI Whisper (local)
- **Audio Extraction**: yt-dlp + ffmpeg
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Groq (openai/gpt-oss-120b)
- **Retrieval**: cosine similarity (scikit-learn)
