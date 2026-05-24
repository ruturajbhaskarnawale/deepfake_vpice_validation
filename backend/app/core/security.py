import hmac
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jose import JWTError, jwt
from backend.app.core.config import settings

api_key_header = APIKeyHeader(name=settings.API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Validates the enterprise API key passed inside the request headers.
    Accepts both the Sentinel platform dev keys AND the NVIDIA API key
    so testers can use their NVIDIA API key directly in Swagger / Playground.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "API Key missing. Provide the 'x-api-key' header. "
                "Use the Sentinel dev key 'sentinel_dev_key_2026_top_secret' "
                "or your NVIDIA API key (nvapi-...)."
            )
        )

    # 1. Check against the configured Sentinel platform dev keys
    for key in settings.API_KEYS:
        if hmac.compare_digest(api_key, key):
            return api_key

    # 2. Also accept the NVIDIA API key from .env (nvapi-...)
    #    so users can authenticate with the same key used for NIM calls.
    if settings.NVIDIA_APIKEY and hmac.compare_digest(api_key, settings.NVIDIA_APIKEY):
        return api_key

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            "Invalid or unauthorized API key. "
            "Use 'sentinel_dev_key_2026_top_secret' or your NVIDIA API key (nvapi-...)."
        )
    )


def create_access_token(data: dict, expires_delta: Optional[int] = None) -> str:
    """
    Utility generating analytical token JWT hashes.
    """
    to_encode = data.copy()
    expire_time = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    if expires_delta:
        expire_time = expires_delta
        
    import datetime
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=expire_time)
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> dict:
    """
    Validates token payload integrity.
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials token."
        )
