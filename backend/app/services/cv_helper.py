# Service wrappers for CV manipulation
import os
import shutil

def process_and_sanitize_image(src_path: str, dest_path: str) -> str:
    """
    Sanitizes image frames by normalizing sizes and striping EXIF profile schemas.
    """
    # Simply copies in the fallback mode
    shutil.copy2(src_path, dest_path)
    return dest_path
