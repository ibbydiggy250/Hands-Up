import os
import requests
import tempfile
import pygame

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Default (Rachel voice)

def speak_text(text):
    """Send text to ElevenLabs API and play it as speech."""
    if not ELEVENLABS_API_KEY:
        print("⚠️ ElevenLabs API key not found — skipping voice output.")
        return

    payload = {
        "text": text,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "accept": "audio/mpeg",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(f"{ELEVENLABS_URL}/{VOICE_ID}", json=payload, headers=headers)
        response.raise_for_status()

        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(response.content)
            audio_path = f.name

        # Play audio
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        pygame.mixer.quit()

    except Exception as e:
        print("ElevenLabs error:", e)
