"""
Speech Processing Toolkit
Alejandro Varela García

This script demonstrates five different ways to work with speech:
1. Local Speech-to-Text (Offline)
2. Local Text-to-Speech (Offline)
3. Cloud Speech-to-Text (API)
4. Cloud Text-to-Speech (API)
5. Speech Emotion Recognition (Extra task beyond STT/TTS)

I designed this to show how we can use both local resources (for privacy/speed)
and cloud APIs (for accuracy) depending on the situation.
"""

import os
import io
import warnings
# I ignore warnings to keep the terminal clean during the demo
warnings.filterwarnings("ignore")

# Task 1 & 2: Local Processing Libraries
# I use Whisper for local STT. It runs entirely on my computer without internet.
# I use pyttsx3 for local TTS. It uses the built-in voices of my OS (Windows/Mac).
try:
    import whisper
    import pyttsx3
except ImportError:
    print("Error: Please install whisper and pyttsx3. Run: pip install openai-whisper pyttsx3")

# Task 3 & 4: Google Cloud API Libraries
# These require an internet connection and a valid API key.
# I put the credentials in a .env file for security.
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as texttospeech
from dotenv import load_dotenv
load_dotenv()  # Load my secret key from .env file

# Task 5: Extra Speech Task (Emotion Recognition)
# I chose emotion recognition because it helps us understand prosody (tone of voice).
# I use a pre-trained model from HuggingFace.
from transformers import pipeline
import librosa