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
        
    # 4. Processing Transcript
    print("\n[*] Processing transcript and generating embeddings...")
    chunks = embedder.chunk_text(transcript, sentences_per_chunk=3)
    if not chunks:
        print("Error: Could not extract any valid sentences from the transcript.")
        sys.exit(0)
        
    stored_data = embedder.embed_chunks(chunks)
    print(f"[*] Transcript was successfully split into {len(chunks)} chunks and stored in-memory.")
    
    # 5. Question/Answer Loop
    while True:
        print("\n" + "="*60)
        query = input("Enter your question (or type 'exit' to quit): ").strip()
        
        if query.lower() in ['exit', 'quit']:
            print("Exiting application. Goodbye!")
            break
            
        if not query:
            continue
            
        print("\n[*] Retrieving relevant chunks...")
        query_emb = embedder.embed_query(query)
        top_results = get_top_k(query_emb, stored_data, top_k=1)
        
        if not top_results:
            print("[-] No relevant context found.")
            continue
            
        # Displaying retrieved context as a bonus step
        print("\n[Retrieved Context]")
        context_texts = []
        for i, res in enumerate(top_results, 1):
            print(f"  Chunk {i} (Similarity: {res['score']:.4f}) | {res['text']}")
            context_texts.append(res['text'])
            
        combined_context = "\n\n".join(context_texts)
        
        # 6. Groq LLM Querying
        prompt = f"""You are a helpful assistant. Answer the question ONLY using the provided context. If the answer is not in the context, say 'I don't know'.

Context:
{combined_context}

Question:
{query}

Answer clearly and concisely."""

        print("\n[Querying Groq LLM...]")
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a precise and helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="openai/gpt-oss-120b",
            )
            answer = chat_completion.choices[0].message.content
            print("\n[Final Answer]")
            print(answer)
            print("="*60)
            
        except Exception as e:
            print(f"\n[-] Error contacting Groq API: {e}")

if __name__ == "__main__":
    main()