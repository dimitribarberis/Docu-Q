from rag.pdf_parser import extract_text_from_pdf
from rag.embedder import chunk_text, embed_chunks

texts = extract_text_from_pdf("data/uploaded_docs/sample_bill.pdf")

chunks = chunk_text(texts, max_chars=500, overlap=50)

print(f"Chunk count: {len(chunks)}\n")
print(f"Example chunk:\n{chunks[0]}\n")
print(f"Example chunk:\n{chunks[1]}\n")

embeddings = embed_chunks(chunks)

print(f"Embeddings shape: {embeddings.shape}\n")