import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


def build_faiss_index(embeddings):
    """
    Builds a FAISS index from the given embeddings.

    Args:
        embeddings (np.ndarray): Array of embeddings to index.

    Returns:
        faiss.IndexFlatL2: FAISS index
            (exact search using euclidean distance)
            (fast, perfect for small amounts of chunks)
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def save_index(index, chunks, index_path="models/faiss_index/index.bin", meta_path="models/faiss_index/chunks.pkl"):
    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(chunks, f)

def load_index(index_path="models/faiss_index/index.bin", meta_path="models/faiss_index/chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve(query, index, chunks, top_k=3):
    """
    Retrieves the top K chunks most similar to the query.

    Args:
        query (str): The query string.
        index (faiss.IndexFlatL2): FAISS index to search.
        chunks (List[str]): List of text chunks corresponding to the embeddings.
        top_k (int): Number of top results to return.

    Returns:
        List[str]: List of tuples containing the chunk and its similarity score.
    """
    query_vec = model.encode([query])
    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]