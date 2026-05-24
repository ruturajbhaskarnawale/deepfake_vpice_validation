import os
import sys

def test_local_whisper():
    print("Testing local Whisper via transformers...")
    try:
        import torch
        from transformers import pipeline
        print("Imported torch and transformers successfully.")
        
        # Load ASR pipeline
        print("Loading whisper-tiny pipeline...")
        # To avoid large downloads, we use openai/whisper-tiny
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
        print("Pipeline loaded successfully.")
        
        wav_path = "test_assets/test_extracted_audio.wav"
        if os.path.exists(wav_path):
             print(f"Transcribing {wav_path}...")
             res = pipe(wav_path)
             print("Result:", res)
        else:
             print("WAV file not found.")
    except Exception as e:
        print("Failed with error:", e)

if __name__ == "__main__":
    test_local_whisper()
