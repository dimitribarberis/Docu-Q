# Docu Q
DocuQ is a lightweight GenAI-powered app that lets you upload utility bills (PDFs) and ask natural language questions like "What was my total electricity usage in April?"

## Features
- PDF parsing and chunking
- Semantic search with FAISS
- Local embedding with Sentence-Transformers
- Local LLM answering (via Ollama or HuggingFace)
- TODO: Streamlit UI for chat-like Q&A

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
4. Upload a utility bill PDF and ask a question!

## Model Options
- Local: [Mistral](https://ollama.com/library/mistral)
- Optional: OpenAI GPT-3.5 (low cost)

## License
MIT
