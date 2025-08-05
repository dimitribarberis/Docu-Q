from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS

def embed_and_store(chunks, persist_path="vectorstore"):
    """
    Embeds chunks and stores them in a FAISS vectorstore.
    - `chunks` = list of dicts with 'text' and 'metadata'
    - `persist_path` = folder to save FAISS index
    """
    # Extract plain texts and metadata
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]

    # Use Ollama embedding model (make sure it's running)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create FAISS vector store
    vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

    # Save locally
    vectorstore.save_local(persist_path)

    print(f"✅ Vector store saved to '{persist_path}'")

def load_vectorstore(persist_path="vectorstore"):
    """
    Loads a FAISS vectorstore from disk.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)