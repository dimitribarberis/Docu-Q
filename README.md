# Docu Q
DocuQ is a lightweight GenAI-powered app that lets you upload PDFs and ask natural language questions.

## Features
- PDF parsing and chunking
- Semantic search with FAISS
- Local embedding with Sentence-Transformers
- Local LLM answering (via Ollama or HuggingFace)
- Streamlit UI for chat-like Q&A

## Quickstart
1. Clone the repo:
```bash
git clone https://github.com/yourusername/docuQ.git
cd docuQ
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the app locally:
```bash
streamlit run app.py
```
4. Upload PDFs and ask a question!

## Model Options
- Local: phi3
- Optional: OpenAI GPT-3.5 (low cost)

## License
MIT
