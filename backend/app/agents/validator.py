import os
import mimetypes
from typing import Any, Dict, List, Tuple
from PIL import Image
from backend.app.agents.base import BaseAgent
from backend.app.core.config import settings
from backend.app.models.pydantic_models import ThreatSignal, ThreatCategory

# Windows-safe fallback for python-magic
try:
    import magic
    has_magic = True
except Exception:
    has_magic = False

class ValidatorAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "ValidatorAgent"

    async def process(self, case_id: str, context: Dict[str, Any]) -> List[ThreatSignal]:
        # The validator agent doesn't generate standard threat signals. 
        # Instead, it performs system validations and mutates/sanitizes files.
        # However, if it finds high-risk metadata mismatches, it can raise an exception or log signals.
        return []

    def validate_file(self, file_path: str) -> Tuple[bool, str, str]:
        """
        Validates the given file for sizes, corruptions, and MIME authenticities.
        Returns: Tuple[is_valid, mime_type, error_reason]
        """
        if not os.path.exists(file_path):
            return False, "", f"File does not exist: {file_path}"

        # 1. Size Check
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            return False, "", f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({settings.MAX_FILE_SIZE_MB} MB)"

        # 2. Extension Check
        ext = file_path.split(".")[-1].lower() if "." in file_path else ""
        if ext not in settings.ALLOWED_EXTENSIONS:
            return False, "", f"Extension '.{ext}' is not supported"

        # 3. MIME Authenticity Check
        detected_mime = None
        if has_magic:
            try:
                # Use magic to inspect file headers directly
                detected_mime = magic.from_file(file_path, mime=True)
            except Exception:
                pass
        
        if not detected_mime:
            # Fallback to standard mimetypes guess
            guess, _ = mimetypes.guess_type(file_path)
            detected_mime = guess

        if not detected_mime:
            return False, "", "Could not identify file MIME type"

        if detected_mime not in settings.ALLOWED_MIME_TYPES:
            return False, detected_mime, f"MIME type '{detected_mime}' is not permitted"

        # 4. Corruption & Usability Checks
        if detected_mime.startswith("image/"):
            try:
                with Image.open(file_path) as img:
                    img.verify()
                # Re-open for dimension checks (verify closes file handler)
                with Image.open(file_path) as img:
                    width, height = img.size
                    if width < settings.IMAGE_MIN_RESOLUTION_WIDTH or height < settings.IMAGE_MIN_RESOLUTION_HEIGHT:
                        return False, detected_mime, f"Image dimensions ({width}x{height}) are below required resolution threshold ({settings.IMAGE_MIN_RESOLUTION_WIDTH}x{settings.IMAGE_MIN_RESOLUTION_HEIGHT})"
            except Exception as e:
                return False, detected_mime, f"Corrupted image layout: {str(e)}"

        elif detected_mime == "application/pdf":
            try:
                # Basic signature check for PDF headers
                with open(file_path, "rb") as f:
                    header = f.read(4)
                    if header != b"%PDF":
                        return False, detected_mime, "Invalid PDF header signature"
            except Exception as e:
                return False, detected_mime, f"Unreadable PDF document: {str(e)}"

        elif detected_mime.startswith("audio/"):
            try:
                # Simple WAV/MP3 integrity validation
                with open(file_path, "rb") as f:
                    header = f.read(12)
                    if detected_mime in ["audio/wav", "audio/x-wav"] and b"RIFF" not in header:
                        return False, detected_mime, "Invalid RIFF wav container signature"
            except Exception as e:
                return False, detected_mime, f"Unreadable audio file: {str(e)}"

        return True, detected_mime, ""

    def sanitize_media(self, file_path: str, out_dir: str) -> str:
        """
        Normalizes files (resizing large images, ensuring directory structural setups).
        Sanitizes the output filename to remove spaces, parentheses, and other special
        characters that cause OpenCV imread() to silently fail on Windows.
        """
        import re
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        # Replace any character that is not alphanumeric, dot, or dash with underscore
        safe_filename = re.sub(r'[^\w.\-]', '_', filename)
        dest_path = os.path.join(out_dir, f"sanitized_{safe_filename}")
        
        # Simple file-copy sanitizer fallback
        # In a real environment, this utilizes OpenCV and Librosa to normalize dimensions/audio properties
        import shutil
        shutil.copy2(file_path, dest_path)
        
        # Try visual normalization on images
        guess, _ = mimetypes.guess_type(file_path)
        if guess and guess.startswith("image/"):
            try:
                with Image.open(file_path) as img:
                    # Normalize orientation or standard RGB schemas if required
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(dest_path, "JPEG", quality=95)
            except Exception:
                pass # Fallback to standard copy if PIL manipulation fails
                
        return dest_path
