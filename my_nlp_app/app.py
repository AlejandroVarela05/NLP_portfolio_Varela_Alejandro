# # Sentiment & Aspect Analyzer - Local LLM Application
# 
# This application uses a local Ollama model to analyse the sentiment of a text
# and extract the main aspects mentioned. It includes preprocessing, prompt engineering,
# and postprocessing to provide structured output.

# Import necessary libraries
import gradio as gr      # For the graphical user interface
import requests          # To communicate with Ollama's API
import re                # For regular expressions (text cleaning)

# Configuration
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "gemma3:1b"   # Lightweight model that works on my laptop

# 1. Preprocessing function
def preprocess_text(text: str) -> str:
    """
    Clean the input text before sending it to the LLM.
    I do this to remove noise and make the prompt clearer.
    """
    # Convert to lowercase (helps the model generalise)
    text = text.lower()
    # Remove extra whitespaces (multiple spaces become one)
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing spaces
    text = text.strip()
    return text

# 2. Prompt engineering
def build_prompt(cleaned_text: str) -> str:
    """
    Create a structured prompt that asks the LLM to output
    sentiment and aspects in a parseable format.
    I use few-shot examples to guide the model.
    """
    prompt = f"""You are a sentiment analysis assistant. Given a user text, you must:
1. Classify the sentiment as Positive, Negative, or Neutral.
2. List the main aspects or topics mentioned (e.g., 'food', 'service', 'price').

Answer strictly in the following format:

Sentiment: <Positive/Negative/Neutral>
Aspects: <comma-separated list of aspects>

Examples:
Text: "The pizza was delicious but the service was slow."
Sentiment: Neutral
Aspects: pizza, service

Text: "Great product, fast shipping!"
Sentiment: Positive
Aspects: product, shipping

Now analyse this text:
Text: "{cleaned_text}"
"""
    return prompt

# 3. Call Ollama and parse the response
def call_ollama(prompt: str):
    """
    Send the prompt to the local Ollama model and return the raw response.
    I set temperature=0 to make the output more deterministic (easier to parse).
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code}"

def postprocess(raw_output: str):
    """
    Extract sentiment and aspects from the LLM's raw output.
    I use simple string searching because the prompt forces a fixed format.
    If parsing fails, I return default values.
    """
    sentiment = "Unknown"
    aspects = []
    
    lines = raw_output.split('\n')
    for line in lines:
        if line.startswith("Sentiment:"):
            sentiment = line.replace("Sentiment:", "").strip()
        elif line.startswith("Aspects:"):
            aspects_str = line.replace("Aspects:", "").strip()
            if aspects_str:
                aspects = [a.strip() for a in aspects_str.split(',')]
    
    # Ensure sentiment is one of the three expected values
    if sentiment not in ["Positive", "Negative", "Neutral"]:
        sentiment = "Unknown"
    
    return sentiment, aspects