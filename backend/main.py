"""
FastAPI backend for AskLecture.
Handles text processing, video transcription, and embedding storage.
"""

import os
import sys
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()  # Load .env file automatically

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding import TranscriptEmbedder
from src.retrieval import get_top_k
from backend.audio import download_audio, cleanup
from backend.whisper_transcribe import transcribe_with_whisper


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
embedder: TranscriptEmbedder | None = None
stored_data: list[dict] = []

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the embedding model once at startup."""
    global embedder
    print("[Backend] Loading sentence-transformer model...")
    embedder = TranscriptEmbedder()
    print("[Backend] Embedding model ready.")
    yield
    print("[Backend] Shutting down.")


# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AskLecture API",
    description="Backend for AskLecture — transcript processing, video transcription, and semantic retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class TextRequest(BaseModel):
    text: str


class VideoRequest(BaseModel):
    video_url: str


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3


class ProcessResponse(BaseModel):
    status: str
    num_chunks: int
    transcript: str | None = None


class RetrieveResult(BaseModel):
    text: str
    score: float


class RetrieveResponse(BaseModel):
    results: list[RetrieveResult]


class HealthResponse(BaseModel):
    status: str
    embeddings_loaded: bool
    num_chunks_stored: int


# ---------------------------------------------------------------------------
# Shared processing pipeline
# ---------------------------------------------------------------------------
def process_text_pipeline(text: str) -> int:
    """
    Core processing function — chunks text, generates embeddings, stores them.
    This is the SINGLE shared pipeline used by both /process-text and /process-video.

    Returns:
        Number of chunks created.
    """
    global stored_data

    if not text or not text.strip():
        raise ValueError("Text is empty.")

    chunks = embedder.chunk_text(text.strip(), sentences_per_chunk=3)
    if not chunks:
        raise ValueError("Could not extract meaningful sentences from the text.")

    stored_data = embedder.embed_chunks(chunks)
    return len(chunks)
