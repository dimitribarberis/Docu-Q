#pdfplumber is slightly easier to use than PyMuPDF
import pdfplumber

def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from each page of a PDF.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[str]: List of page-wise extracted text.
    """
    page_texts = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                #clean weird spacing or empty lines
                cleaned_text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
                page_texts.append(cleaned_text)
            else:
                page_texts.append(f"Page {i + 1} has no extractable text.")

    return page_texts