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
