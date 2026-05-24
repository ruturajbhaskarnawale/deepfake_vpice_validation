import os
import sys
import httpx

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from backend.app.core.config import settings

def get_all_models():
    url = "https://integrate.api.nvidia.com/v1/models"
    headers = {
        "Authorization": f"Bearer {settings.NVIDIA_APIKEY}"
    }
    response = httpx.get(url, headers=headers)
    if response.status_code == 200:
        models = response.json().get("data", [])
        print("Total models:", len(models))
        for m in sorted([x["id"] for x in models]):
            print(" -", m)
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    get_all_models()
