import os
import sys

def test_wav2vec2():
    print("Testing local Wav2Vec2 via transformers...")
    try:
        from transformers import pipeline
        print("Imported pipeline.")
        pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
        print("Wav2Vec2 Pipeline loaded successfully!")
        
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
    test_wav2vec2()
