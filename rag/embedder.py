from sentence_transformers import SentenceTransformer
import textwrap

model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(pages, max_chars=500, overlap=50):
    """
    Splits each page of text into smaller overlapping chunks to preserve some context.

    Args:
        pages (List[str]): List of page-wise text content.
        max_chars (int): Maximum characters per chunk.
        overlap (int): Number of overlapping characters between chunks.µ

    Returns:
        List[str]: List of text chunks.
    """

    chunks = []

    for page in pages:
        if not page:
            continue
        start = 0
        while start < len(page):
            end = min(start + max_chars, len(page))
            chunk = page[start:end]
            chunks.append(chunk.strip())
            start += max_chars - overlap

    return chunks

def embed_chunks(chunks):
    """
    Embeds text chunks using a pre-trained SentenceTransformer model.

    Args:
        chunks (List[str]): List of text chunks to embed.

    Returns:
        np.ndarray: Embeddings matrix.
    """

    embeddings = model.encode(chunks)
    return embeddings