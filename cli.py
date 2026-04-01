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

    # 3. Inputting the Transcript
    print("\n--- Document Semantic Search CLI ---")
    print("Please paste your transcript below.")
    print("Type 'DONE' on a new line when finished.\n")
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == 'DONE':
                break
            lines.append(line)
        except EOFError:
            break
            
    transcript = "\n".join(lines).strip()
    
    if not transcript:
        print("Error: Empty transcript provided. Exiting.")
        sys.exit(0)
    
   
if __name__ == "__main__":
    main()