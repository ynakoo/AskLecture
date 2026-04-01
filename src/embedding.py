import re
from sentence_transformers import SentenceTransformer

class TranscriptEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the sentence transformer model."""
        self.model = SentenceTransformer(model_name)
    
    def clean_text(self, text: str) -> str:
        """Removes extra spaces and line breaks from the text."""
        # Replace newlines and multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def chunk_text(self, text: str, sentences_per_chunk: int = 3) -> list[str]:
        """Splits cleaned text into chunks of roughly `sentences_per_chunk` sentences."""
        text = self.clean_text(text)
        
        # Simple sentence splitting on punctuation (. ! ?)
        # Using regex to split keeping punctuation intact.
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        # Group sentences into chunks
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = " ".join(sentences[i:i+sentences_per_chunk])
            chunks.append(chunk)
            
        return chunks
    def embed_chunks(self, chunks: list[str]) -> list[dict]:
        """Creates embeddings for a list of text chunks and stores them in memory."""
        if not chunks:
            return []
            
        embeddings = self.model.encode(chunks)
        
        stored_data = []
        for text, emb in zip(chunks, embeddings):
            stored_data.append({
                "text": text,
                "embedding": emb
            })
            
        return stored_data
    def embed_query(self, query: str):
        """Creates an embedding for a single user query."""
        # encode() returns a 2D array if list is provided, take the first item
        return self.model.encode([query])[0]