import os
import sys
import asyncio
import base64
import httpx

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from backend.app.core.config import settings

async def describe_file(file_path):
    print(f"\n--- Describing {file_path} ---")
    if not os.path.exists(file_path):
        print("File not found.")
        return
        
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        
        mime_type = "image/png" if file_path.endswith(".png") else "image/jpeg"
        data_url = f"data:{mime_type};base64,{encoded_string}"

        headers = {
            "Authorization": f"Bearer {settings.NVIDIA_APIKEY}",
            "Content-Type": "application/json"
        }
        
        prompt = (
            "Describe this image in detail. If it is an ID card or document, extract all text and visible details."
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
            "max_tokens": 1024,
            "temperature": 0.2
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
        
        if response.status_code == 200:
            print(response.json()["choices"][0]["message"]["content"])
        else:
            print("Error:", response.text)
    except Exception as e:
        print("Failed with error:", e)

async def main():
    await describe_file("test_assets/Image (7).jpg")
    await describe_file("test_assets/Image (8).jpg")
    await describe_file("test_assets/Media.jpg")

if __name__ == "__main__":
    asyncio.run(main())
