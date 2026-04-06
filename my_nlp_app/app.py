# # Sentiment & Aspect Analyzer - Local LLM Application
# 
# This application uses a local Ollama model to analyse the sentiment of a text
# and extract the main aspects mentioned. It includes preprocessing, prompt engineering,
# and postprocessing to provide structured output.

# %%
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

