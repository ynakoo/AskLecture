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
        