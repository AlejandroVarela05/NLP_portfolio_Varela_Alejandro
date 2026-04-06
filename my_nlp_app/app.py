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

# 4. Main pipeline (preprocess -> prompt -> call -> postprocess)
def analyze_sentiment(user_text: str):
    """
    This is the function that Gradio will call when the user clicks the button.
    It ties all the steps together and returns a formatted result.
    """
    if not user_text.strip():
        return "Please enter some text.", "No aspects found."
    
    # Step 1: The Preprocess
    cleaned = preprocess_text(user_text)
    
    # Step 2: Build prompt
    prompt = build_prompt(cleaned)
    
    # Step 3: Call LLM
    raw_answer = call_ollama(prompt)
    
    # Step 4: Postprocess
    sentiment, aspects = postprocess(raw_answer)
    
    # Format output for the GUI
    result_text = f"**Sentiment:** {sentiment}\n\n**Aspects:** {', '.join(aspects) if aspects else 'None'}"
    # Also return the raw response for debugging
    debug_info = f"*(Raw LLM output)*\n{raw_answer}"
    
    return result_text, debug_info

# 5. Build the Gradio interface
# I choose Gradio because it's very easy to create a web UI with minimal code.
# The interface has:
#   - A textbox for user input
#   - A button to trigger analysis
#   - Two output boxes: one for the structured result, one for raw output
with gr.Blocks(title="Sentiment & Aspect Analyzer") as demo:
    gr.Markdown("#Sentiment & Aspect Analyzer")
    gr.Markdown("Enter a sentence or a product review, and the local LLM will tell you the sentiment and the main aspects mentioned.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Your text", lines=5, placeholder="e.g., The food was amazing but the wait was too long.")
            analyze_btn = gr.Button("Analyze")
        with gr.Column():
            output_result = gr.Markdown(label="Result")
            output_debug = gr.Textbox(label="Raw LLM output (for inspection)", lines=6)
    
    analyze_btn.click(fn=analyze_sentiment, inputs=input_text, outputs=[output_result, output_debug])

# 6. Launch the app
if __name__ == "__main__":
    # Launch with share=False (local only) but you can set share=True for a temporary public link
    demo.launch(server_name="127.0.0.1", server_port=7860)