import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"

def ask_llm(prompt):
    """
    Sends a prompt to the local Ollama LLM and returns the generated answer.

    Args:
        prompt (str): The prompt to send to the LLM.

    Returns:
        str: The LLM's generated response.
    """

    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False  # Set to True if you want streaming responses
    })

    if response.status_code == 200:
        return response.json()['response'].strip()
    else:
        print("LLM call failed:", response.status_code, response.text)
        return "Error: Unable to get response from LLM."
