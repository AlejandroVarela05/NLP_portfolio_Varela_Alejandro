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
The user can record their own voice in real time to test the system interactively.
"""

import os
import io
import warnings
import time

# I ignore warnings to keep the terminal clean during the demo
warnings.filterwarnings("ignore")

# Get the directory where this script is located
# All generated files will be saved here.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Task 1 & 2: Local Processing Libraries
try:
    import whisper
    import pyttsx3
except ImportError:
    print("Error: Please install whisper and pyttsx3. Run: pip install openai-whisper pyttsx3")

# Task 3: AssemblyAI Cloud API (Speech-to-Text)
import assemblyai as aai
from dotenv import load_dotenv
load_dotenv()  # Load my secret key from .env file

# Task 4: gTTS Cloud API (Text-to-Speech)
from gtts import gTTS

# Task 5: Extra Task (Emotion Recognition)
from transformers import pipeline
import librosa

# Audio Recording (for live demo)
try:
    import pyaudio
    import wave
    LIVE_RECORDING_AVAILABLE = True
except ImportError:
    LIVE_RECORDING_AVAILABLE = False
    print("[WARNING] PyAudio not installed. Live recording disabled.")
    print("Install with: pip install pipwin && pipwin install pyaudio")


# 0. RECORD AUDIO FROM MICROPHONE
def record_audio(duration=5, samplerate=16000, filename="recorded_audio.wav"):
    """
    Record audio from the default microphone using PyAudio.
    """
    if not LIVE_RECORDING_AVAILABLE:
        print("[RECORD] ERROR: Live recording is not available.")
        return None

    save_path = os.path.join(SCRIPT_DIR, filename)
    chunk = 1024
    audio_format = pyaudio.paInt16
    channels = 1

    p = pyaudio.PyAudio()

    print(f"[RECORD] Preparing to record for {duration} seconds...")
    print("[RECORD] Speak clearly into the microphone.")

    for i in range(3, 0, -1):
        print(f"[RECORD] {i}...")
        time.sleep(1)
    print("[RECORD] Recording NOW!")

    stream = p.open(format=audio_format,
                    channels=channels,
                    rate=samplerate,
                    input=True,
                    frames_per_buffer=chunk)

    frames = []
    for _ in range(0, int(samplerate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("[RECORD] Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save as WAV file
    wf = wave.open(save_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(samplerate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"[RECORD] Audio saved to {save_path}")
    return save_path


# 1. LOCAL SPEECH TO TEXT (Whisper)
def local_speech_to_text(audio_path):
    """
    Transcribe an audio file using a local Whisper model.
    """
    print(f"[LOCAL STT] Loading Whisper model ('base')...")
    model = whisper.load_model("base")
    
    print(f"[LOCAL STT] Transcribing {audio_path}...")
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
    """
    save_path = os.path.join(SCRIPT_DIR, save_path)  
    print(f"[LOCAL TTS] Initializing engine...")
    engine = pyttsx3.init()
    
    print(f"[LOCAL TTS] Generating speech for: '{text[:50]}...'")
    engine.save_to_file(text, save_path)
    engine.runAndWait()
    print(f"[LOCAL TTS] Audio saved to {save_path}")
    return save_path


# 3. EXTERNAL API SPEECH TO TEXT (AssemblyAI)
def assemblyai_speech_to_text(audio_path):
    """
    Transcribe audio using AssemblyAI's cloud API.
    """
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("[CLOUD STT] ERROR: ASSEMBLYAI_API_KEY not found in .env file.")
        return ""
    
    aai.settings.api_key = api_key
    
    print(f"[CLOUD STT] Uploading and transcribing '{audio_path}' with AssemblyAI...")
    print("[CLOUD STT] This may take a few seconds...")
    
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(
        language_detection=True,
        punctuate=True,
        format_text=True
    )
    
    try:
        transcript = transcriber.transcribe(audio_path, config=config)
        
        if transcript.status == aai.TranscriptStatus.error:
            print(f"[CLOUD STT] Error: {transcript.error}")
            return ""
        
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
    """
    save_path = os.path.join(SCRIPT_DIR, save_path)
    print(f"[CLOUD TTS] Connecting to Google TTS servers...")
    
    if not text or len(text.strip()) == 0:
        print("[CLOUD TTS] ERROR: No text provided.")
        return None

    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        print(f"[CLOUD TTS] Synthesizing speech for: '{text[:50]}...'")
        tts.save(save_path)
        print(f"[CLOUD TTS] Audio saved to {save_path}")
        return save_path
    except Exception as e:
        print(f"[CLOUD TTS] Error: {e}")
        return None


# 5. EXTRA TASK: SPEECH EMOTION RECOGNITION (Local)
def detect_emotion_from_speech(audio_path):
    """
    Detect the emotion in a spoken utterance.
    """
    print(f"[EXTRA TASK] Loading emotion recognition model...")
    try:
        classifier = pipeline("audio-classification", model="superb/wav2vec2-large-superb-er")
    except Exception as e:
        print(f"[EXTRA TASK] Error loading model: {e}")
        return None
    
    print(f"[EXTRA TASK] Processing audio file: {audio_path}")
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
    
    result = classifier(audio_array)
    top_emotion = result[0]['label']
    confidence = result[0]['score']
    
    print(f"[EXTRA TASK] Detected emotion: {top_emotion} (Confidence: {confidence:.2f})")
    return top_emotion


# 6. LIVE DEMO MODE (One-click full pipeline)
def live_demo_mode():
    """
    This function runs a fully automated demo.
    The user only needs to press Enter and speak.
    All five tasks are executed automatically on the recorded audio.
    """
    print("  LIVE DEMO MODE - AUTOMATIC SPEECH PIPELINE")
    print("\nThis mode will record your voice and then run ALL tasks automatically.")
    print("You don't need to select anything else.")
    print("\nPress ENTER when you are ready to speak...")
    input()
    
    # Record for 5 seconds
    recorded_file = record_audio(duration=5, filename="live_demo_audio.wav")
    if recorded_file is None:
        print("[DEMO] Live recording failed. Please check your microphone and PyAudio installation.")
        return
    
    print("  PROCESSING YOUR SPEECH...")
    
    # 1. Local STT
    print("\n1. LOCAL SPEECH-TO-TEXT (Whisper)")
    text_local = local_speech_to_text(recorded_file)
    
    # 2. Cloud STT
    print("\n2. CLOUD SPEECH-TO-TEXT (AssemblyAI)")
    text_cloud = assemblyai_speech_to_text(recorded_file)
    
    # 3. Emotion recognition
    print("\n3. EMOTION RECOGNITION (Extra Task)")
    emotion = detect_emotion_from_speech(recorded_file)
    
    # 4. Local TTS (using the transcribed text)
    print("\n4. LOCAL TEXT-TO-SPEECH (pyttsx3)")
    if text_local:
        local_text_to_speech(text_local, "local_tts_output.wav")
    else:
        print("[LOCAL TTS] No transcription available to synthesize.")
    
    # 5. Cloud TTS (using the transcribed text)
    print("\n5. CLOUD TEXT-TO-SPEECH (gTTS)")
    if text_cloud:
        cloud_text_to_speech(text_cloud, "cloud_tts_output.mp3")
    else:
        print("[CLOUD TTS] No transcription available to synthesize.")
    
    # Summary
    print("  DEMO COMPLETE - SUMMARY")
    print(f"You said (local):  {text_local}")
    print(f"You said (cloud):  {text_cloud}")
    print(f"Emotion detected: {emotion}")
    print("\nAudio files generated:")
    print("  - local_tts_output.wav (robotic voice)")
    print("  - cloud_tts_output.mp3 (natural voice)")
    print("\nThank you for watching the demo!")


# MAIN INTERFACE (Interactive Menu)
def main():
    """
    This menu makes it easy to record the demonstration video.
    I can quickly test each function one by one or run the automatic live demo.
    """
    print("  SPEECH TOOLKIT - FINAL PROJECT DEMO")
    
    sample_wav = os.path.join(SCRIPT_DIR, "sample_audio.wav")
    recorded_wav = os.path.join(SCRIPT_DIR, "recorded_audio.wav")
    sample_text = "Hello, this is a test of the speech processing toolkit. I am demonstrating local and cloud capabilities."
    
    while True:
        print("\nChoose an option:")
        print("0. Record new audio (speak into microphone)")
        print("1. Local Speech-to-Text (Whisper)")
        print("2. Local Text-to-Speech (pyttsx3)")
        print("3. Cloud Speech-to-Text (AssemblyAI)")
        print("4. Cloud Text-to-Speech (gTTS)")
        print("5. Emotion Recognition (Extra Task)")
        print("6. Run Full Demo Sequence (with recorded audio)")
        print("7. LIVE DEMO MODE (Automatic - speak once, see all)")
        print("8. Exit")
        
        choice = input("> ").strip()
        
        # Option 0: Record audio
        if choice == "0":
            dur = input("Enter recording duration in seconds (default 5): ")
            if not dur:
                dur = 5
            else:
                dur = int(dur)
            record_audio(duration=dur, filename="recorded_audio.wav")
            print("[INFO] Recorded audio is now available as 'recorded_audio.wav'.")
            continue
        
        # For options 1,3,5 we ask which audio file to use
        audio_to_use = None
        if choice in ["1", "3", "5"]:
            print("Which audio file to use?")
            print("  a) Pre-recorded sample (sample_audio.wav)")
            print("  b) Last recorded audio (recorded_audio.wav)")
            print("  c) Enter custom path")
            sub = input("> ").strip().lower()
            if sub == "a":
                audio_to_use = sample_wav
            elif sub == "b":
                audio_to_use = recorded_wav
            elif sub == "c":
                audio_to_use = input("Enter full path: ").strip()
            else:
                print("Invalid choice. Using sample_audio.wav by default.")
                audio_to_use = sample_wav
            
            if not os.path.exists(audio_to_use):
                print(f"[ERROR] File {audio_to_use} does not exist.")
                continue
        
        # Option 1: Local STT
        if choice == "1":
            local_speech_to_text(audio_to_use)
        
        # Option 2: Local TTS
        elif choice == "2":
            txt = input("Enter text to speak (or press Enter for default): ")
            if not txt:
                txt = sample_text
            local_text_to_speech(txt)
        
        # Option 3: Cloud STT (AssemblyAI)
        elif choice == "3":
            assemblyai_speech_to_text(audio_to_use)
        
        # Option 4: Cloud TTS (gTTS)
        elif choice == "4":
            txt = input("Enter text to speak (or press Enter for default): ")
            if not txt:
                txt = sample_text
            cloud_text_to_speech(txt)
        
        # Option 5: Emotion Recognition
        elif choice == "5":
            detect_emotion_from_speech(audio_to_use)
        
        # Option 6: Full Demo Sequence (using recorded audio)
        elif choice == "6":
            print("\nRUNNING FULL DEMO SEQUENCE")
            if not os.path.exists(recorded_wav):
                print("[DEMO] No recorded audio found. Let's record one first (5 seconds).")
                record_audio(duration=5, filename="recorded_audio.wav")
            else:
                print(f"[DEMO] Using existing recorded audio: {recorded_wav}")
            
            print("\n[1] Local STT (Whisper):")
            text_local = local_speech_to_text(recorded_wav)
            
            print("\n[2] Local TTS (pyttsx3) - generating speech from default text:")
            local_text_to_speech(sample_text)
            
            print("\n[3] Cloud STT (AssemblyAI):")
            text_cloud = assemblyai_speech_to_text(recorded_wav)
            
            print("\n[4] Cloud TTS (gTTS) - generating speech from default text:")
            cloud_text_to_speech(sample_text)
            
            print("\n[5] Emotion Recognition:")
            emotion = detect_emotion_from_speech(recorded_wav)
            
            print("\n DEMO COMPLETE")
            print(f"Your spoken words (local): {text_local}")
            print(f"Your spoken words (cloud): {text_cloud}")
            print(f"Detected emotion: {emotion}")
        
        # Option 7: LIVE DEMO MODE
        elif choice == "7":
            live_demo_mode()
        
        # Option 8: Exit
        elif choice == "8":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select a number from the menu.")

if __name__ == "__main__":
    main()