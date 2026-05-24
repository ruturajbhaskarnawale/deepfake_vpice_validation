import os
import sys
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from backend.app.core.config import settings

def test_openai_sdk():
    print("Testing Whisper using OpenAI SDK...")
    
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=settings.NVIDIA_APIKEY
    )
    
    wav_path = "test_assets/test_extracted_audio.wav"
    if not os.path.exists(wav_path):
        print(f"WAV file not found at {wav_path}.")
        return

    try:
        # Let's try different model names
        for model in ["nvidia/whisper-large-v3", "meta/whisper-large-v3", "whisper-large-v3"]:
            print(f"Trying model: {model}")
            try:
                with open(wav_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model=model,
                        file=audio_file
                    )
                print(f"Success with {model}!")
                print("Transcript:", transcription.text)
                return
            except Exception as e:
                print(f"Failed with {model}: {e}")
    except Exception as e:
        print("General error:", e)

if __name__ == "__main__":
    test_openai_sdk()
