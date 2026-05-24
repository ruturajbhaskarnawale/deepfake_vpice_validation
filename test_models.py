import os
import sys
import asyncio
import logging
import base64
import httpx
from PIL import Image

# Force backend/ onto PYTHONPATH
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from backend.app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_models")

async def test_paddleocr():
    print("\n--- Testing PaddleOCR ---")
    try:
        from paddleocr import PaddleOCR
        print("PaddleOCR is importable.")
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        print("PaddleOCR initialized successfully.")
        
        # Test on a dummy image or one of the test assets
        test_img = "test_assets/Image (7).jpg"
        if os.path.exists(test_img):
            print(f"Running PaddleOCR on {test_img}...")
            result = ocr.ocr(test_img, cls=True)
            print("PaddleOCR result count:", len(result) if result else 0)
            if result and result[0]:
                for line in result[0][:5]:
                    print("  OCR Line:", line[1][0])
        else:
            print(f"Test image {test_img} not found.")
    except Exception as e:
        print("PaddleOCR test failed with error:", e)

async def test_nvidia_vision():
    print("\n--- Testing NVIDIA NIM Vision (meta/llama-3.2-11b-vision-instruct) ---")
    if not settings.NVIDIA_APIKEY:
        print("NVIDIA_APIKEY is not set.")
        return
        
    test_img = "test_assets/Image (7).jpg"
    if not os.path.exists(test_img):
        print(f"Test image {test_img} not found.")
        return
        
    try:
        with open(test_img, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        
        mime_type = "image/jpeg"
        data_url = f"data:{mime_type};base64,{encoded_string}"

        headers = {
            "Authorization": f"Bearer {settings.NVIDIA_APIKEY}",
            "Content-Type": "application/json"
        }
        
        prompt = (
            "You are an expert AI forensic analyst. Analyze this document scan:\n"
            "Return a raw JSON block containing keys 'full_name', 'document_type', 'date_of_birth', 'gender'."
        )

        payload = {
            "model": "meta/llama-3.2-11b-vision-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ],
            "max_tokens": 512,
            "temperature": 0.2
        }

        print("Sending request to NVIDIA NIM Vision API...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
        
        print("NVIDIA NIM Vision Response Status:", response.status_code)
        if response.status_code == 200:
            print("Response content:")
            print(response.json()["choices"][0]["message"]["content"])
        else:
            print("Response error:", response.text)
    except Exception as e:
        print("NVIDIA NIM Vision test failed with error:", e)

async def test_nvidia_whisper():
    print("\n--- Testing NVIDIA NIM Whisper (meta/whisper-large-v2) ---")
    if not settings.NVIDIA_APIKEY:
        print("NVIDIA_APIKEY is not set.")
        return
        
    # We can try to use a real WAV file or mock WAV file
    test_audio = "test_assets/MicrosoftTeams-video.mp4" # Whisper needs audio, let's see if we have one
    # Wait, let's see if there is an extracted audio WAV from a previous run or we can extract one.
    # In test_assets, we don't have WAV, but let's see if we have one in sanitized/ or we can generate a small mock WAV or test directly.
    # Wait, can we extract from MicrosoftTeams-video.mp4? Let's check.
    temp_wav = "test_assets/test_extracted_audio.wav"
    if os.path.exists(test_audio):
        try:
            import subprocess
            print("Extracting audio from video...")
            subprocess.run([
                "ffmpeg", "-y", "-i", test_audio, "-vn",
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_wav
            ], capture_output=True)
            print("Audio extracted successfully.")
        except Exception as e:
            print("ffmpeg extraction failed:", e)
            
    if not os.path.exists(temp_wav):
        print("No WAV file to test Whisper.")
        return
        
    try:
        with open(temp_wav, "rb") as f:
            audio_bytes = f.read()
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        data_url = f"data:audio/wav;base64,{b64_audio}"

        headers = {
            "Authorization": f"Bearer {settings.NVIDIA_APIKEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "meta/whisper-large-v2",
            "messages": [
                {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": data_url}}]}
            ],
            "max_tokens": 512,
            "temperature": 0.0,
        }

        print("Sending request to NVIDIA NIM Whisper API...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            
        print("NVIDIA NIM Whisper Response Status:", response.status_code)
        if response.status_code == 200:
            print("Response content:")
            print(response.json()["choices"][0]["message"]["content"])
        else:
            print("Response error:", response.text)
    except Exception as e:
        print("NVIDIA NIM Whisper test failed with error:", e)
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

async def main():
    print("NVIDIA_APIKEY:", settings.NVIDIA_APIKEY[:10] + "..." if settings.NVIDIA_APIKEY else "None")
    await test_paddleocr()
    await test_nvidia_vision()
    await test_nvidia_whisper()

if __name__ == "__main__":
    asyncio.run(main())
