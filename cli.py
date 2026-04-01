import os
import sys
from groq import Groq
from src.embedding import TranscriptEmbedder
from src.retrieval import get_top_k

def main():
     # 1. Verification of the API Key
    api_key = "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"
    client = Groq(api_key=api_key)
    
    # 2. Embedding Model Setup
    print("[*] Initializing embedding model ('all-MiniLM-L6-v2')...")
    embedder = TranscriptEmbedder()
    
   
if __name__ == "__main__":
    main()