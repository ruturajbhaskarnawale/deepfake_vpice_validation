import os
import sys
import asyncio
import subprocess
import httpx

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from backend.app.core.config import settings

def extract_audio():
    video_path = "test_assets/MicrosoftTeams-video.mp4"
    out_path = "test_assets/test_extracted_audio.wav"
    ffmpeg_bin = os.path.abspath("backend/bin/ffmpeg.exe")
    if not os.path.exists(ffmpeg_bin):
         ffmpeg_bin = "ffmpeg"
         
    print(f"Using ffmpeg: {ffmpeg_bin}")
    cmd = [
        ffmpeg_bin, "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_path
    ]
    subprocess.run(cmd, capture_output=True)
    print(f"Extracted to {out_path}, size={os.path.getsize(out_path) if os.path.exists(out_path) else 0}")
    return out_path

async def test_whisper_call(model_name):
    print(f"\n--- Testing Whisper with model: {model_name} ---")
    wav_path = "test_assets/test_extracted_audio.wav"
    if not os.path.exists(wav_path):
        print("WAV file not found.")
        return
        
    url = "https://integrate.api.nvidia.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {settings.NVIDIA_APIKEY}"
    }
    
    files = {
        "file": ("audio.wav", open(wav_path, "rb"), "audio/wav")
    }
    data = {
        "model": model_name
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, files=files, data=data)
        print("Status code:", response.status_code)
        if response.status_code == 200:
            print("Response:", response.json())
        else:
            print("Error response:", response.text)
    except Exception as e:
        print("Exception:", e)

async def main():
    extract_audio()
    # Test typical models in the catalog
    await test_whisper_call("nvidia/whisper-large-v3")
    await test_whisper_call("meta/whisper-large-v3")
    await test_whisper_call("whisper-large-v3")

if __name__ == "__main__":
    asyncio.run(main())
