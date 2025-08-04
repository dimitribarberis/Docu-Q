def build_prompt(question, context_chunks):
    """
    Build a prompt for the LLM using context chunks and the question.

    Args:
        question (str): The question to be answered.
        context_chunks (List[str]): List of context chunks to provide additional information.

    Returns:
        str: Final prompt string.
    """
    context = "\n---\n".join(context_chunks)

    prompt = f"""You are an assistant that helps users understand their utility bills.
    
    Use only the information below to answer the question. If the answer isn't found, say you don't know.
    
    Context:
    {context}
    Question: {question}
    Answer:"""

    return prompt
