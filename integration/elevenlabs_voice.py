import os
import requests
import tempfile
import pygame


ELEVENLABS_API_KEY = "sk_f42cc07cf78caaf11c70d615b6b45d24306e5f246f62b19c"
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel

def speak_text(text):
    """Send text to ElevenLabs API and play it as speech."""
    if not ELEVENLABS_API_KEY:
        print("⚠️ ElevenLabs API key not found — skipping voice output.")
        return

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
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

        # Save temporary MP3 file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(response.content)
            audio_path = f.name

        # Play audio
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.quit()
        os.remove(audio_path)

    except requests.exceptions.RequestException as e:
        print("❌ ElevenLabs API error:", e)
    except pygame.error as e:
        print("🎵 Pygame audio error:", e)
    except Exception as e:
        print("⚠️ Unexpected error:", e)

if __name__ == "__main__":
    speak_text("Hello! This is HandsUp speaking using ElevenLabs.")