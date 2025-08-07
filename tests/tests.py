# test_pipeline.py
from rag.pdf_parser import extract_text_from_pdf, chunk_text
from rag.embedder import embed_and_store, load_vectorstore
from rag.qa_engine import query_legal_doc

def full_pipeline_test(pdf_path, test_question):
    print("Step 1: Extracting text from PDF...")
    pages = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(pages)} pages.")

    print("Step 2: Chunking document...")
    chunks = chunk_text(pages)
    print(f"Created {len(chunks)} chunks.")

    print("Step 3: Embedding and storing in FAISS...")
    embed_and_store(chunks)
    print("Vectorstore saved.")

    print("Step 4: Asking your question...")
    answer = query_legal_doc(test_question)
    print("\n🔎 Question:", test_question)
    print("Answer:", answer)


if __name__ == "__main__":
    # Example usage
    pdf_path = "data/uploaded_docs/sample_court.pdf"
    test_question = "What does this document describe?"

    full_pipeline_test(pdf_path, test_question)
