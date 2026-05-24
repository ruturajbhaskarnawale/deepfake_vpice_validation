import os
import sys
import asyncio
import base64
import httpx

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from backend.app.core.config import settings

async def test_omni():
    print("Testing Nemotron Omni audio transcription...")
    wav_path = "test_assets/test_extracted_audio.wav"
    if not os.path.exists(wav_path):
        print("WAV file not found.")
        return
        
    try:
        with open(wav_path, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")
            
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.NVIDIA_APIKEY}",
            "Content-Type": "application/json"
        }
        
        prompt = (
            "You are a transcription assistant. Transcribe the spoken audio exactly. "
            "Do not add any explanations or introductory text, just return the transcription."
        )
        
        payload = {
            "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_audio,
                                "format": "wav"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.0
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
        print("Status code:", response.status_code)
        if response.status_code == 200:
            print("Response content:")
            print(response.json()["choices"][0]["message"]["content"])
        else:
            print("Error:", response.text)
    except Exception as e:
        print("Exception:", e)

if __name__ == "__main__":
    asyncio.run(test_omni())
