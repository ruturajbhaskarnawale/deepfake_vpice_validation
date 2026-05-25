import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from backend.app.models.pydantic_models import ThreatSignal

class BaseAgent(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """The logical name of this agent."""
        pass

    @abstractmethod
    async def process(self, case_id: str, context: Dict[str, Any]) -> List[ThreatSignal]:
        """Processes the given case context and returns a list of detected ThreatSignals."""
        pass

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """
        Extracts and parses a JSON object from text content.
        Tries direct parsing, code block extraction, and brace-matching extraction.
        """
        if not content:
            return {}

        content_str = content.strip()
        
        # Try direct parse
        try:
            return json.loads(content_str)
        except json.JSONDecodeError:
            pass

        # Try matching ```json ... ```
        match = re.search(r"```json\s*(.*?)\s*```", content_str, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try matching ``` ... ``` without language
        match = re.search(r"```\s*(.*?)\s*```", content_str, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try finding the first '{' and last '}'
        start = content_str.find('{')
        end = content_str.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = content_str[start:end+1]
            try:
                return json.loads(candidate.strip())
            except json.JSONDecodeError:
                pass

        return {}

    async def _call_nvidia_nim_with_fallback(self, stage: str, headers: dict, payload: dict) -> tuple[Any | None, str | None]:
        """
        Executes a NIM API request for the specified pipeline stage.
        If the primary model fails or isn't configured, falls back to the secondary model.
        Returns the successful response and the model name used, or (None, None) if both failed.
        """
        import httpx
        import asyncio
        import logging
        from backend.app.core.config import settings
        
        agent_logger = logging.getLogger(f"sentinel.{self.name.lower()}")
        
        stage_cfg = settings.MODELS.get(stage, {})
        primary_model = stage_cfg.get("primary")
        secondary_model = stage_cfg.get("secondary")
        
        models_to_try = []
        if primary_model:
            models_to_try.append(primary_model)
        if secondary_model:
            models_to_try.append(secondary_model)
            
        # If no models defined for this stage, fall back to general defaults
        if not models_to_try:
            if stage in ("voice_authenticity", "speech_to_text"):
                models_to_try = [settings.MODELS["nvidia_nim"]["audio_model"]]
            else:
                models_to_try = [settings.MODELS["nvidia_nim"]["vlm_model"]]
                
        last_error = None
        for model in models_to_try:
            payload["model"] = model
            agent_logger.info(f"Attempting NIM API call using model '{model}' for stage '{stage}'...")
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response = await client.post(
                            "https://integrate.api.nvidia.com/v1/chat/completions",
                            headers=headers,
                            json=payload
                        )
                    if response.status_code == 200:
                        agent_logger.info(f"NIM API call succeeded with model '{model}'")
                        return response, model
                    
                    agent_logger.warning(f"NIM API error ({response.status_code}) on model '{model}' (attempt {attempt + 1}): {response.text}")
                    last_error = f"HTTP {response.status_code}: {response.text}"
                except httpx.HTTPError as exc:
                    agent_logger.warning(f"HTTP connection error on model '{model}' (attempt {attempt + 1}): {str(exc)}")
                    last_error = str(exc)
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                    
        agent_logger.error(f"All configured NIM models for stage '{stage}' failed. Last error: {last_error}")
        return None, None

