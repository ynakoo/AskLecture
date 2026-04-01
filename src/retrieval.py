import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_top_k(query_embedding: np.ndarray, stored_data: list[dict], top_k: int = 3) -> list[dict]:
    """
    Computes cosine similarity between the query embedding and all stored chunk embeddings.
    Returns the top_k most similar chunks.
    
    Args:
        query_embedding (np.ndarray): 1D array representing the user question embedding.
        stored_data (list[dict]): The in-memory storage of text chunks and their embeddings.
        top_k (int): Number of most relevant chunks to return.
        
    Returns:
        list[dict]: List containing top_k relevant chunk dictionaries with 'text' and 'score'.
    """
    if not stored_data:
        return []
        
    # Extract all embeddings into a 2D numpy array
    chunk_embeddings = np.array([item["embedding"] for item in stored_data])
    
    # Reshape query embedding from 1D to 2D for sklearn function (1, number_of_features)
    query_vec = query_embedding.reshape(1, -1)
    
    # Compute similarity: returns shape (1, num_chunks)
    similarities = cosine_similarity(query_vec, chunk_embeddings)[0]
    
    # Get indices of the top-k highest similarity scores (argsort returns ascending)
    # We take the last `top_k` and reverse the array to get descending order
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Format and return the results
    results = []
    for idx in top_indices:
        results.append({
            "text": stored_data[idx]["text"],
            "score": float(similarities[idx])
        })
        
    return results