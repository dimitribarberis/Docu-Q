from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from rag.embedder import load_vectorstore

llm = Ollama(model="phi3")

template = """
You are a helpful assistant specialized in understanding documents and answering questions about them.
Answer the question below using ONLY the provided context.

If the answer isn't in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:
"""

prompt = PromptTemplate.from_template(template)

def query_legal_doc(question, k=4):
    """
    Query the document using the question and return the answer and source documents.
    """
    # Load the vectorstore
    vectorstore = load_vectorstore()

    # Retrieve top-k relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    # Format context from chunks
    context = ""
    for doc in docs:
        context += f"{doc.page_content.strip()}\n---\n"

    final_prompt = prompt.format(context=context, question=question)

    answer = llm.invoke(final_prompt)

    return answer, docs
