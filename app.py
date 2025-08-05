import streamlit as st
from rag.pdf_parser import extract_text_from_pdf, chunk_documents
from rag.embedder import embed_and_store, load_vectorstore
from rag.qa_engine import query_legal_doc
import os

st.set_page_config(page_title="LegalDoc QA", page_icon="🔍")
st.title("🔍 Legal Document Q&A")

# --- File Upload ---
st.header("1. Upload your legal documents (PDF)")
uploaded_files = st.file_uploader("Upload one or more legal PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("✅ Process & Index PDFs"):
        all_chunks = []
        for uploaded_file in uploaded_files:
            with open(f"data/uploaded_docs/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getvalue())

            pages = extract_text_from_pdf(f"data/uploaded_docs/{uploaded_file.name}")
            chunks = chunk_documents(pages, filename=uploaded_file.name)
            all_chunks.extend(chunks)

        embed_and_store(all_chunks)
        st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} documents.")

# --- Question Interface ---
st.header("2. Ask a question")

templates = {
    "🔍 Ask a custom question": None,
    "📑 Summarize this document": "Please summarize the key points in this document.",
    "🧾 What are the obligations of the tenant?": "List the obligations of the tenant as stated in the document.",
    "❌ What are the termination conditions?": "Explain the termination conditions mentioned in this document.",
}
selected = st.selectbox("Choose a question template", list(templates.keys()))
if selected == "🔍 Ask a custom question":
    question = st.text_input("Enter your question")
else:
    question = templates[selected]

if question:
    with st.spinner("Thinking..."):
        answer, sources = query_legal_doc(question)

        st.markdown("### 🤖 Answer")
        st.write(answer)

        st.markdown("### 📚 Sources")
        for doc in sources:
            filename = doc.metadata.get("filename", "Unknown file")
            page = doc.metadata.get("page", "Unknown page")
            preview = doc.page_content[:300].strip()

            st.markdown(f"**📄 {filename} — Page {page}**")
            st.code(preview)

