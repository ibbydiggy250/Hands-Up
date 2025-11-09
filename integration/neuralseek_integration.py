import os
import requests

# Read your API key from environment variable
NEURALSEEK_API_KEY = os.getenv("NEURALSEEK_API_KEY")
NEURALSEEK_URL = "https://api.neuralseek.com/v1/rephrase"

def improve_sentence(raw_text: str) -> str:
    """
    Sends the short ASL label to NeuralSeek and returns
    a more natural English sentence.
    """
    if not NEURALSEEK_API_KEY:
        # fallback if key not set
        return raw_text.replace("_", " ").capitalize()

    payload = {
        "input": raw_text.replace("_", " "),
        "goal": "Convert ASL phrase into a natural English sentence suitable for subtitles."
    }
    headers = {
        "Authorization": f"Bearer {NEURALSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(NEURALSEEK_URL, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("output", raw_text)
        else:
            print("NeuralSeek API error:", response.status_code, response.text)
            return raw_text
    except Exception as e:
        print("NeuralSeek request failed:", e)
        return raw_text
