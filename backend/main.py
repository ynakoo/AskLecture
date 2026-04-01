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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        embeddings_loaded=embedder is not None,
        num_chunks_stored=len(stored_data),
    )


@app.post("/process-text", response_model=ProcessResponse)
async def process_text(request: TextRequest):
    """
    Process pasted transcript text.
    Chunks the text, generates embeddings, and stores them for retrieval.
    """
    try:
        num_chunks = process_text_pipeline(request.text)
        return ProcessResponse(status="processed", num_chunks=num_chunks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@app.post("/process-video", response_model=ProcessResponse)
async def process_video(request: VideoRequest):
    """
    Process a video URL:
    1. Download audio using yt-dlp
    2. Transcribe with local Whisper model
    3. Feed transcript into the same text processing pipeline
    """
    audio_path = None

    try:
        # Step 1: Download audio
        print(f"[Backend] Downloading audio from: {request.video_url}")
        audio_path = download_audio(request.video_url)
        print(f"[Backend] Audio saved to: {audio_path}")

        # Step 2: Transcribe with Whisper
        print(f"[Backend] Transcribing with Whisper ({WHISPER_MODEL_SIZE})...")
        transcript = transcribe_with_whisper(audio_path, model_size=WHISPER_MODEL_SIZE)
        print(f"[Backend] Transcription complete. Length: {len(transcript)} chars.")

        # Step 3: Process through shared pipeline (same as /process-text)
        num_chunks = process_text_pipeline(transcript)

        return ProcessResponse(
            status="processed",
            num_chunks=num_chunks,
            transcript=transcript,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")
    finally:
        # Step 4: Always clean up audio file
        if audio_path:
            cleanup(audio_path)
            print(f"[Backend] Cleaned up temp audio file.")


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """
    Retrieve the top-k most relevant chunks for a given query.
    The Groq LLM call is handled by the Streamlit frontend.
    """
    if not stored_data:
        raise HTTPException(
            status_code=400,
            detail="No transcript has been processed yet. Please process a transcript first.",
        )

    try:
        query_emb = embedder.embed_query(request.query)
        top_results = get_top_k(query_emb, stored_data, top_k=request.top_k)

        return RetrieveResponse(
            results=[
                RetrieveResult(text=r["text"], score=r["score"])
                for r in top_results
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
