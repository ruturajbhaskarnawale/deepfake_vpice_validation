import os
import sys
import asyncio
from paddleocr import PaddleOCR

def test_ocr_on_assets():
    print("--- Running OCR on test_assets/ ---")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    
    for filename in ["Image (7).jpg", "Image (8).jpg", "Media.jpg"]:
        path = os.path.join("test_assets", filename)
        if os.path.exists(path):
            print(f"\nOCR on {filename}:")
            result = ocr.ocr(path, cls=True)
            if result and result[0]:
                for line in result[0]:
                    print("  - Text:", line[1][0], "| Conf:", line[1][1])
            else:
                print("  No text found.")
        else:
            print(f"\n{filename} does not exist.")

if __name__ == "__main__":
    test_ocr_on_assets()
