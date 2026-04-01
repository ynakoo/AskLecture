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