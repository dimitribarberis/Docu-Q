from rag.pdf_parser import extract_text_from_pdf
from rag.embedder import chunk_text, embed_chunks
from rag.retriever import build_faiss_index, save_index, load_index, retrieve
from rag.llm_interface import ask_llm
from rag.prompt_utils import build_prompt

"""
EXTRACTION TEST
__________________________________________________
"""
texts = extract_text_from_pdf("data/uploaded_docs/sample_bill.pdf")

"""
__________________________________________________
CHUNCKING AND EMBEDDING EXAMPLE
__________________________________________________
"""

chunks = chunk_text(texts, max_chars=300, overlap=30)

print(f"Chunk count: {len(chunks)}\n")
print(f"Example chunk:\n{chunks[0:4]}\n")

embeddings = embed_chunks(chunks)

print(f"Embeddings shape: {embeddings.shape}\n")

"""
___________________________________________________
RETRIEVAL TEST
___________________________________________________
"""
index = build_faiss_index(embeddings)
#save_index(index, chunks)

#index, chunks = load_index()
#results = retrieve("What is the total amount of the bill?", index, chunks)

"""
------------------------------------------------
LLM RESPONSE TEST
------------------------------------------------
"""
question = ("What is the combined amount due from metered and unmetered usage? You can find the actual numbers for usage in kwh and effective rate in section 2 and the unmetered amount in section 3. Therefor you have to find a numerical answer!")
top_chunks = retrieve(question, index, chunks)

prompt = build_prompt(question, top_chunks)
answer = ask_llm(prompt)

print("\n💬 Question:", question)
print("🧠 Answer:", answer)