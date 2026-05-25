import os
import json
from typing import List, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

def load_models_list() -> Dict[str, Any]:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "models_list.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {
        "nvidia_nim": {
            "vlm_model": "meta/llama-3.2-11b-vision-instruct",
            "audio_model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        },
        "local": {
            "face_biometric_model_buffalo": "buffalo_l",
            "face_liveness_model": "SilentFace-Liveness",
            "face_matching_model": "InsightFace-ArcFace",
            "video_liveness_model": "MediaPipe-Liveness",
            "video_deepfake_model": "TimeSformer-XCLIP",
            "audio_authenticity_model": "AASIST-v2-Spectral"
        },
        "ocr": {
            "primary": "meta/llama-3.2-11b-vision-instruct",
            "secondary": "nvidia/neva-22b",
            "fallback": "local_heuristic_ocr"
        },
        "vision_forensics": {
            "primary": "meta/llama-3.2-11b-vision-instruct",
            "secondary": "nvidia/neva-22b",
            "fallback": "SilentFace-Liveness"
        },
        "voice_authenticity": {
            "primary": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            "secondary": "nvidia/nemotron-4-340b-instruct",
            "fallback": "AASIST-v2-Spectral"
        },
        "speech_to_text": {
            "primary": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            "secondary": "nvidia/whisper-large-v3",
            "fallback": "local_dummy_transcription"
        },
        "identity_graph": {
            "primary": "meta/llama-3.2-11b-vision-instruct",
            "secondary": "nvidia/nemotron-4-340b-instruct",
            "fallback": "local_graph_service_heuristics"
        },
        "risk_scorer": {
            "primary": "meta/llama-3.2-11b-vision-instruct",
            "secondary": "nvidia/nemotron-4-340b-instruct",
            "fallback": "local_rule_scorer_heuristics"
        }
    }

class Settings(BaseSettings):
    MODELS: Dict[str, Any] = load_models_list()
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General Project Info
    PROJECT_NAME: str = "Jodetx Sentinel Core"
    API_V1_STR: str = "/api/v1"
    
    # Security
    JWT_SECRET_KEY: str = Field(default="SUPER_SECRET_JODETX_TOKEN_KEY_CHANGE_ME_IN_PRODUCTION", alias="JWT_SECRET")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    API_KEY_NAME: str = "x-api-key"
    API_KEYS: List[str] = ["sentinel_dev_key_2026_top_secret"]
    
    # NVIDIA API Key config
    NVIDIA_APIKEY: str = Field(default="", alias="NVIDIA_APIKEY")

    # Storage & Relational Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://sentinel_user:sentinel_password@localhost:5432/sentinel_db"
    )
    
    # Redis Queue & Cache
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # Ingestion Validation Configs
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = [
        "pdf", "jpg", "jpeg", "png", "mp4", "webm", "wav", "mp3"
    ]
    ALLOWED_MIME_TYPES: List[str] = [
        "application/pdf",
        "image/jpeg",
        "image/png",
        "video/mp4",
        "video/webm",
        "audio/webm",
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3"
    ]

    # Model Quality Thresholds
    IMAGE_MIN_RESOLUTION_WIDTH: int = 100
    IMAGE_MIN_RESOLUTION_HEIGHT: int = 100
    AUDIO_MIN_SAMPLING_RATE_HZ: int = 16000

settings = Settings()
