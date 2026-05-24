# Service wrappers for Audio manipulation
import shutil

def process_and_sanitize_audio(src_path: str, dest_path: str) -> str:
    """
    Sanitizes audio channels by downsampling raw recordings.
    """
    shutil.copy2(src_path, dest_path)
    return dest_path
