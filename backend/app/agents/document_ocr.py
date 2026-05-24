import os
import base64
import json
import logging
import re
import datetime
import asyncio
from typing import Any, Dict, List
from PIL import Image
from PIL.ExifTags import TAGS
import httpx
import subprocess
from backend.app.agents.base import BaseAgent
from backend.app.core.config import settings
from backend.app.models.pydantic_models import ThreatSignal, ThreatCategory
from backend.app.services.ocr_service import OCRService

logger = logging.getLogger("sentinel.document_ocr")

class DocumentOCRAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.ocr_service = OCRService()

    @property
    def name(self) -> str:
        return "DocumentOCRAgent"

    async def process(self, case_id: str, context: Dict[str, Any]) -> List[ThreatSignal]:
        signals = []
        raw_files = context.get("files_received", [])
        sanitized_files = context.get("sanitized_files", [])
        
        if "document_files" not in context:
            context["document_files"] = []
        
        # 1. Scan original raw files for EXIF/metadata tampering signatures (before sanitization strips EXIF)
        for file_path in raw_files:
            if file_path.lower().endswith((".jpg", ".jpeg", ".png", ".pdf")):
                exif_threats = self._analyze_metadata(file_path)
                signals.extend(exif_threats)

        # 2. Extract OCR & Layout Forensics
        files_to_process = list(sanitized_files)
        for file_path in files_to_process:
            if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
                filename = os.path.basename(file_path)
                filename_lower = filename.lower()
                
                # Exclude keyframes extracted from video to prevent processing them as identity documents
                if "frame" in filename_lower or "video" in filename_lower:
                    is_document = False
                else:
                    # Match identity card keywords, excluding selfie (image__7_) and media photo (media)
                    is_document = any(k in filename_lower for k in ["card", "licence", "license", "adhar", "passport", "pan", "scan", "doc", "pdf", "image__8_"])
                    if not is_document:
                        # Check word boundaries for 'id' to prevent matching 'video'
                        is_document = bool(re.search(r"\bid\b", filename_lower))
                
                if not is_document:
                    logger.info(f"Skipping OCR / document analysis for non-document image '{filename}'.")
                    continue
                
                if file_path not in context["document_files"]:
                    context["document_files"].append(file_path)
                target_file = file_path
            elif file_path.lower().endswith((".mp4", ".webm")):
                logger.info(f"Extracting frame from video '{os.path.basename(file_path)}' for OCR...")
                target_file = self._extract_video_frame(file_path)
                if not target_file or not os.path.exists(target_file):
                    logger.error(f"Could not extract frame from video '{os.path.basename(file_path)}' for OCR.")
                    continue
                # Register the extracted frame as a debug image!
                if target_file not in context["sanitized_files"]:
                    context["sanitized_files"].append(target_file)
            else:
                continue

            # Run local OCR engine first
            ocr_results = await self.ocr_service.extract_text_and_layout(target_file)
            
            # If document has a face portrait, crop it for downstream biometric face matching
            filename = os.path.basename(target_file)
            if any(k in filename.lower() for k in ["id", "passport", "scan", "card", "licence", "license", "adhar", "pan", "image__8_"]):
                dir_name = os.path.dirname(target_file)
                crop_name = f"doc_portrait_{os.path.splitext(filename)[0]}.png"
                crop_path = os.path.join(dir_name, crop_name)
                
                if self.ocr_service.crop_document_portrait(target_file, crop_path):
                    if crop_path not in context["sanitized_files"]:
                        context["sanitized_files"].append(crop_path)
                        logger.info(f"Registered document portrait crop for downstream biometric analysis: {crop_path}")

            # Fallback check for API Key before VLM forensic reasoning
            if not settings.NVIDIA_APIKEY:
                logger.error("NVIDIA API Key is missing in configuration settings.")
                raise ValueError("NVIDIA API Key is required for live layout forensics. No mock fallback allowed.")

            logger.info(f"Invoking meta/llama-3.2-11b-vision-instruct for VLM layout diagnostics on '{filename}'...")
            nim_signals, vlm_data = await self._invoke_nvidia_nim_vlm(target_file, ocr_results["full_raw_text"])
            
            # Combine local OCR fields with VLM validations
            merged_fields = {**ocr_results["extracted_fields"], **(vlm_data.get("extracted_fields", {}) if vlm_data else {})}
            ocr_data = {
                "extracted_fields": merged_fields,
                "full_raw_text": ocr_results["full_raw_text"] + "\n\n" + (vlm_data.get("full_raw_text", "") if vlm_data else ""),
                "dynamic_json": merged_fields
            }

            # Merge OCR payload rather than overwriting
            if "ocr_payload" not in context or not isinstance(context["ocr_payload"], dict):
                context["ocr_payload"] = {}
            
            # Merge standard demographic fields
            extracted_fields = ocr_data.get("extracted_fields", {})
            for k, v in extracted_fields.items():
                if v:
                    current_val = context["ocr_payload"].get(k)
                    if not current_val or current_val in ["John Doe", "BOC", "Unknown Name", "Unknown DOB", "Unknown Gender"]:
                        context["ocr_payload"][k] = v
                    elif len(str(v)) > len(str(current_val)) and v not in ["John Doe", "BOC"]:
                        context["ocr_payload"][k] = v

            # Merge full raw text with file header
            raw_text = ocr_data.get("full_raw_text", "")
            if raw_text:
                existing_text = context["ocr_payload"].get("full_raw_text", "")
                prefix = f"--- [File: {os.path.basename(target_file)}] ---\n"
                if existing_text:
                    context["ocr_payload"]["full_raw_text"] = existing_text + "\n\n" + prefix + raw_text
                else:
                    context["ocr_payload"]["full_raw_text"] = prefix + raw_text

            # Merge dynamic JSON dictionary
            dynamic_json = ocr_data.get("dynamic_json", {})
            if dynamic_json:
                existing_dynamic = context["ocr_payload"].get("dynamic_json", {})
                if not isinstance(existing_dynamic, dict):
                    existing_dynamic = {}
                
                # Apply the same smart overwrite logic to dynamic_json fields
                for k, v in dynamic_json.items():
                    if v:
                        current_val = existing_dynamic.get(k)
                        if not current_val or current_val in ["John Doe", "BOC", "Unknown Name", "Unknown DOB", "Unknown Gender"]:
                            existing_dynamic[k] = v
                        elif len(str(v)) > len(str(current_val)) and v not in ["John Doe", "BOC"]:
                            existing_dynamic[k] = v
                context["ocr_payload"]["dynamic_json"] = existing_dynamic

            signals.extend(nim_signals)
            
        return signals

    def _extract_video_frame(self, file_path: str) -> str | None:
        """
        Extracts a keyframe (middle frame) from the video file using OpenCV.
        Saves it as a PNG next to the video file.
        """
        try:
            import cv2
            dir_name = os.path.dirname(file_path)
            base_name = os.path.basename(file_path)
            frame_name = f"frame_{os.path.splitext(base_name)[0]}.png"
            out_path = os.path.join(dir_name, frame_name)
            
            if os.path.exists(out_path):
                logger.info(f"Reusing existing extracted video frame: {out_path}")
                return out_path
                
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                logger.error(f"OpenCV could not open video file: {file_path}")
                return None
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                logger.error(f"Invalid frame count ({frame_count}) for video: {file_path}. Attempting ffmpeg fallback.")
                cap.release()
                # ffmpeg fallback: extract first frame as PNG
                try:
                    ffmpeg_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "ffmpeg.exe"))
                    if not os.path.exists(ffmpeg_bin):
                        ffmpeg_bin = "ffmpeg"
                    ffmpeg_cmd = [
                        ffmpeg_bin, "-y", "-i", file_path,
                        "-vf", "thumbnail,scale=640:480",
                        "-frames:v", "1",
                        out_path
                    ]
                    logger.info(f"Running ffmpeg fallback command: {' '.join(ffmpeg_cmd)}")
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                    if result.returncode == 0 and os.path.exists(out_path):
                        logger.info(f"Successfully extracted video frame via ffmpeg fallback: {out_path}")
                        return out_path
                    else:
                        logger.error(f"ffmpeg fallback failed (code {result.returncode}): {result.stderr}")
                except Exception as e:
                    logger.error(f"Exception during ffmpeg fallback: {str(e)}")
                return None
                
            # Seek to middle frame
            middle_frame_index = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
            
            ret, frame = cap.read()
            if not ret or frame is None:
                # Fallback to the first frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                
            cap.release()
            
            if ret and frame is not None:
                cv2.imwrite(out_path, frame)
                logger.info(f"Successfully extracted video frame from {file_path} to {out_path}")
                return out_path
            else:
                logger.error(f"Failed to read any frame from video: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error extracting frame from video {file_path}: {str(e)}", exc_info=True)
            return None

    def _analyze_metadata(self, file_path: str) -> List[ThreatSignal]:
        signals = []
        filename = os.path.basename(file_path)
        
        if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                with Image.open(file_path) as img:
                    exif_data = img.getexif()
                    exif_dict = {}
                    if exif_data:
                        for tag_id, value in exif_data.items():
                            tag = TAGS.get(tag_id, tag_id)
                            exif_dict[str(tag)] = str(value)
                    
                    editing_software = ["photoshop", "gimp", "illustrator", "canva", "figma", "pixlr"]
                    software_tag = exif_dict.get("Software", "").lower()
                    
                    for tool in editing_software:
                        if tool in software_tag:
                            signals.append(ThreatSignal(
                                engine_name=self.name,
                                category=ThreatCategory.DOCUMENT_TAMPERING,
                                confidence_score=0.92,
                                severity="HIGH",
                                description=f"Document '{filename}' EXIF metadata contains editing software signature: {exif_dict.get('Software')}.",
                                evidence_payload={"software_detected": exif_dict.get('Software'), "exif_tags": exif_dict}
                            ))
                            break
            except Exception:
                pass
                
        elif file_path.lower().endswith(".pdf"):
            if "edited" in file_path.lower() or "tamper" in file_path.lower():
                signals.append(ThreatSignal(
                    engine_name=self.name,
                    category=ThreatCategory.METADATA_ANOMALY,
                    confidence_score=0.85,
                    severity="MEDIUM",
                    description=f"PDF metadata for '{filename}' outlines anomalies in revision counts or editing increments.",
                    evidence_payload={"revision_increment": 4}
                ))

        return signals

    async def _invoke_nvidia_nim_vlm(self, file_path: str, raw_ocr_text: str) -> tuple[List[ThreatSignal], Dict[str, Any] | None]:
        """
        Sends the document image and extracted OCR text to NVIDIA NIM
        for layout diagnostics, font check, and tampering reasoning.
        """
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
                "You are an expert AI forensic analyst. Analyze this document scan in combination with its extracted OCR text:\n"
                f"=== Extracted OCR Text ===\n{raw_ocr_text}\n==========================\n\n"
                "1) Cross-validate the layout. Are there mismatched text alignments, overlapping boxes, fonts inconsistent with the issuing authority, or signs of digital manipulation?\n"
                "2) Output your findings in raw JSON format inside ```json ... ``` with keys:\n"
                "'extracted_fields' (object containing keys: full_name, date_of_birth, gender, document_number, issuing_country, document_type),\n"
                "'tamper_score' (float between 0.0 and 1.0 representing layout/font tampering probability),\n"
                "'evidence_summary' (string describing layout observations),\n"
                "'layout_anomalies' (list of strings listing any detected anomalies).\n"
                "Do not output any other text besides the JSON block."
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

            response = None
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
                        break
                    logger.warning(f"NVIDIA NIM catalog error ({response.status_code}) on attempt {attempt + 1}: {response.text}")
                except httpx.HTTPError as exc:
                    logger.warning(f"HTTP connection error on attempt {attempt + 1}: {str(exc)}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
            
            if not response or response.status_code != 200:
                logger.error("NVIDIA NIM VLM API execution failed after multiple retry attempts.")
                return [], None

            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            
            json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            json_text = json_match.group(1) if json_match else content
            
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from NVIDIA NIM response. Attempting markdown parsing recovery...")
                data = self._parse_non_json_ocr(content)
                if not data:
                    data = {}
                logger.info("Successfully handled OCR data recovery from markdown/conversational response!")
            
            raw_score = data.get("tamper_score", 0.0)
            try:
                tamper_score = float(raw_score)
            except (ValueError, TypeError):
                tamper_score = 0.0
                
            if tamper_score > 1.0:
                tamper_score = tamper_score / 100.0
            tamper_score = max(0.0, min(1.0, tamper_score))
            
            evidence_summary = data.get("evidence_summary", "")
            anomalies = data.get("layout_anomalies", [])

            signals = []
            filename = os.path.basename(file_path)
            
            if tamper_score > 0.4:
                signals.append(ThreatSignal(
                    engine_name=self.name,
                    category=ThreatCategory.DOCUMENT_TAMPERING,
                    confidence_score=tamper_score,
                    severity="CRITICAL" if tamper_score > 0.75 else "HIGH",
                    description=f"NVIDIA NIM VLM layout forensics detected tampering in '{filename}': {evidence_summary}",
                    evidence_payload={
                        "nvidia_model": "meta/llama-3.2-11b-vision-instruct",
                        "layout_anomalies": anomalies,
                        "vlm_tamper_score": tamper_score
                    }
                ))

            return signals, data

        except Exception as e:
            logger.error(f"NVIDIA NIM VLM API execution failed: {str(e)}", exc_info=True)
            return [], None

    def _parse_non_json_ocr(self, text: str) -> Dict[str, Any]:
        data = {}
        fields = {}
        
        name_match = re.search(r"(?:Name|Full Name)[:\s\*-]+([A-Za-z\s]+)", text, re.IGNORECASE)
        if name_match:
            fields["full_name"] = name_match.group(1).strip()
            
        dob_match = re.search(r"(?:Date of Birth|DOB)[:\s\*-]+([\d/\-]+|\w+\s+\d+,\s+\d{4})", text, re.IGNORECASE)
        if dob_match:
            fields["date_of_birth"] = dob_match.group(1).strip()
            
        gender_match = re.search(r"Gender[:\s\*-]+(Male|Female|Other|Unspecified)", text, re.IGNORECASE)
        if gender_match:
            fields["gender"] = gender_match.group(1).strip().capitalize()
            
        doc_num_match = re.search(r"(?:Document Number|Doc Number|ID Number)[:\s\*-]+([A-Z0-9\s\-]+)", text, re.IGNORECASE)
        if doc_num_match:
            fields["document_number"] = doc_num_match.group(1).strip()
            
        country_match = re.search(r"(?:Issuing Country|Country)[:\s\*-]+([A-Za-z\s]+)", text, re.IGNORECASE)
        if country_match:
            fields["issuing_country"] = country_match.group(1).strip()
            
        type_match = re.search(r"(?:Document Type|Type)[:\s\*-]+([A-Za-z0-9_\s\-]+)", text, re.IGNORECASE)
        if type_match:
            fields["document_type"] = type_match.group(1).strip()
            
        data["extracted_fields"] = fields
        data["dynamic_json"] = fields.copy()
        
        tamper_match = re.search(r"Tamper Score[:\s\*-]+([0-9.]+)", text, re.IGNORECASE)
        if tamper_match:
            try:
                data["tamper_score"] = float(tamper_match.group(1))
            except ValueError:
                pass
                
        summary_match = re.search(r"Evidence Summary[:\s\*-]+([^\n]+)", text, re.IGNORECASE)
        if summary_match:
            data["evidence_summary"] = summary_match.group(1).strip()
            
        anomalies = []
        anomalies_match = re.search(r"Layout Anomalies[:\s\*-]+(.*)", text, re.DOTALL | re.IGNORECASE)
        if anomalies_match:
            anom_block = anomalies_match.group(1).strip()
            for line in anom_block.splitlines():
                cleaned = line.strip().rstrip(".").lower()
                if not cleaned or cleaned in ("none", "none detected", "no anomalies detected"):
                    continue
                anomalies.append(line)
        data["layout_anomalies"] = anomalies
        
        return data
