#pdfplumber is slightly easier to use than PyMuPDF
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter


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

def chunk_documents(pages, filename):
    chunks = []
    for i, page_dict in enumerate(pages):
        page_text = page_dict["text"]
        page_num = page_dict.get("metadata").get("page",-1)
        for chunk in RecursiveCharacterTextSplitter(
            chunk_size=500,  # Adjust size as needed
            chunk_overlap=100,  # Overlap for better context
        ).split_text(page_text):
            chunks.append({
                "text": chunk,
                "metadata": {
                    "filename": filename,
                    "page": page_num
                }
            })
    return chunks