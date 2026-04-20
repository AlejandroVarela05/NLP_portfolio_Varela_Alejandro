"""
Speech Processing Toolkit
Alejandro Varela Garcia

This script demonstrates five different ways to work with speech:
1. Local Speech-to-Text (Offline) with OpenAI Whisper
2. Local Text-to-Speech (Offline) with pyttsx3
3. Cloud Speech-to-Text (API) with AssemblyAI
4. Cloud Text-to-Speech (API) with gTTS
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

# Task 3: AssemblyAI Cloud API (Speech-to-Text)
# AssemblyAI provides a generous free tier and no credit card is needed to start.
# I store the API key in a .env file so it never appears in the code.
import assemblyai as aai
from dotenv import load_dotenv
load_dotenv()  # Load my secret key from .env file

# Task 4: gTTS Cloud API (Text-to-Speech)
# I use gTTS (Google Text-to-Speech) because it is free, requires no API key,
# and produces very natural-sounding speech.
from gtts import gTTS

# Task 5: Extra Speech Task (Emotion Recognition)
# I chose emotion recognition because it helps us understand prosody (tone of voice).
# I use a pre-trained model from HuggingFace.
from transformers import pipeline
import librosa

# 1. LOCAL SPEECH TO TEXT (Whisper)
def local_speech_to_text(audio_path):
    """
    Transcribe an audio file using a local Whisper model.
    
    Why I did this:
    - Whisper is open-source and runs offline. Good for privacy.
    - I chose the 'base' model: it's a balance between speed and accuracy.
    - If the audio is in Spanish, it automatically detects it.
    """
    print(f"[LOCAL STT] Loading Whisper model ('base')...")
    # Load the model. 'base' is ~140MB. It's downloaded once and cached.
    model = whisper.load_model("base")
    
    print(f"[LOCAL STT] Transcribing {audio_path}...")
    # The transcribe function does the heavy lifting: it resamples audio, extracts features,
    # and runs the neural network.
    result = model.transcribe(audio_path)
    
    detected_lang = result["language"]
    text = result["text"]
    
    print(f"[LOCAL STT] Detected language: {detected_lang}")
    print(f"[LOCAL STT] Transcription: {text}")
    return text

# 2. LOCAL TEXT TO SPEECH (pyttsx3)
def local_text_to_speech(text, save_path="local_output.wav"):
    """
    Convert text to speech using the computer's built-in voices.
    
    Why I did this:
    - It is completely offline and very fast.
    - I save the result to a WAV file so the user can hear it later.
    - The voice quality is robotic but useful for testing text processing.
    """
    print(f"[LOCAL TTS] Initializing engine...")
    engine = pyttsx3.init()
    
    # I can change the voice or speed here if I want.
    # voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[1].id)  # 0=Male, 1=Female (depends on OS)
    
    print(f"[LOCAL TTS] Generating speech for: '{text[:50]}...'")
    engine.save_to_file(text, save_path)
    engine.runAndWait()
    print(f"[LOCAL TTS] Audio saved to {save_path}")
    return save_path


# 3. EXTERNAL API SPEECH TO TEXT (AssemblyAI)
def assemblyai_speech_to_text(audio_path):
    """
    Transcribe audio using AssemblyAI's cloud API.
    
    Why I did this:
    - AssemblyAI offers a generous free tier ($50 credit).
    - No credit card needed for the trial.
    - I store the API key in a .env file so it never appears in the code.
    - The SDK handles upload, polling, and errors automatically.
    """
    # I load the API key from the environment (set in .env file)
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("[CLOUD STT] ERROR: ASSEMBLYAI_API_KEY not found in .env file.")
        print("[CLOUD STT] Please create a .env file with: ASSEMBLYAI_API_KEY=your_key_here")
        return ""
    
    # Configure the global settings for AssemblyAI
    aai.settings.api_key = api_key
    
    print(f"[CLOUD STT] Uploading and transcribing '{audio_path}' with AssemblyAI...")
    print("[CLOUD STT] This may take a few seconds depending on file size...")
    
    # Create a transcriber object
    transcriber = aai.Transcriber()
    
    # I add some useful configuration options
    config = aai.TranscriptionConfig(
        language_detection=True,   # Automatically detect the spoken language
        punctuate=True,            # Add punctuation and capitalization
        format_text=True           # Clean up formatting (numbers, dates, etc.)
    )
    
    try:
        # Send the file to AssemblyAI. The SDK uploads it automatically.
        transcript = transcriber.transcribe(audio_path, config=config)
        
        # Check if there was an error
        if transcript.status == aai.TranscriptStatus.error:
            print(f"[CLOUD STT] Error: {transcript.error}")
            return ""
        
        # Extract the detected language and transcription text
        detected_lang = transcript.json_response.get("language_code", "unknown")
        text = transcript.text
        
        print(f"[CLOUD STT] Detected language: {detected_lang}")
        print(f"[CLOUD STT] Transcription: {text}")
        return text
        
    except Exception as e:
        print(f"[CLOUD STT] An unexpected error occurred: {e}")
        return ""


# 4. EXTERNAL API TEXT TO SPEECH (gTTS)
def cloud_text_to_speech(text, save_path="cloud_output.mp3", lang="en"):
    """
    Convert text to speech using Google Text-to-Speech (gTTS).
    
    Why I did this:
    - gTTS is free and does not need any API key or credit card.
    - It produces high-quality, natural-sounding MP3 files.
    - It supports many languages, which makes my project more versatile.
    """
    print(f"[CLOUD TTS] Connecting to Google TTS servers...")
    
    # Check if there is text to process
    if not text or len(text.strip()) == 0:
        print("[CLOUD TTS] ERROR: No text provided to synthesize.")
        return None

    try:
        # Create a gTTS object. I can specify the language and whether to speak slowly.
        tts = gTTS(text=text, lang=lang, slow=False)
        
        print(f"[CLOUD TTS] Synthesizing speech for: '{text[:50]}...'")
        # Save the generated speech as an MP3 file.
        tts.save(save_path)
        
        print(f"[CLOUD TTS] Audio saved to {save_path}")
        return save_path
        
    except Exception as e:
        print(f"[CLOUD TTS] An unexpected error occurred: {e}")
        return None
    
    
# 5. EXTRA TASK: SPEECH EMOTION RECOGNITION (Local)
def detect_emotion_from_speech(audio_path):
    """
    Detect the emotion in a spoken utterance.
    
    Why I did this (Beyond STT/TTS):
    - This task is about understanding 'how' someone speaks, not 'what' they say.
    - It uses a pre-trained model on the RAVDESS dataset (acted emotions).
    - I use HuggingFace 'pipeline' to make it easy.
    - The model expects 16kHz audio. I use librosa to load and resample it.
    """
    print(f"[EXTRA TASK] Loading emotion recognition model...")
    # I use 'superb/wav2vec2-large-superb-er' which is a fine-tuned wav2vec2 model.
    # It classifies 6 emotions: neutral, happy, sad, angry, fearful, disgust, surprised.
    try:
        classifier = pipeline("audio-classification", model="superb/wav2vec2-large-superb-er")
    except Exception as e:
        print(f"[EXTRA TASK] Error loading model: {e}")
        return None
    
    print(f"[EXTRA TASK] Processing audio file: {audio_path}")
    # Load audio at 16000 Hz (required by the model)
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
    
    # The pipeline expects a dict or array
    result = classifier(audio_array)
    
    # The result is a list of dicts. I will take the top emotion.
    top_emotion = result[0]['label']
    confidence = result[0]['score']
    
    print(f"[EXTRA TASK] Detected emotion: {top_emotion} (Confidence: {confidence:.2f})")
    return top_emotion


