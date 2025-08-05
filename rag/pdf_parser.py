#pdfplumber is slightly easier to use than PyMuPDF
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from each page of a PDF.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        documents : List of page-wise extracted text.
    """
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                documents.append({
                    "text": page_text,
                    "metadata": {"page": i + 1}
                })
    return documents

def chunk_text(documents, chunk_size=500, chunk_overlap=50):
    """
    Splits each page of text into smaller overlapping chunks to preserve some context.

    Args:
        pages (List[str]): List of page-wise text content.
        max_chars (int): Maximum characters per chunk.
        overlap (int): Number of overlapping characters between chunks.µ

    Returns:
        List[str]: List of text chunks.
    """
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []

    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]

        chunks = splitter.create_documents([text])
        for chunk in chunks:
            all_chunks.append({
                "text": chunk.page_content,
                "metadata": metadata
            })

    return all_chunks