import os
from typing import List, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
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
