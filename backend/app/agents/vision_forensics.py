import os
import base64
import json
import logging
import re
import hashlib
import asyncio
from typing import Any, Dict, List
from PIL import Image
import numpy as np
import httpx
from backend.app.agents.base import BaseAgent
from backend.app.core.config import settings
from backend.app.models.pydantic_models import ThreatSignal, ThreatCategory
from backend.app.services.biometric_service import BiometricService
from backend.app.services.video_service import VideoService

logger = logging.getLogger("sentinel.vision_forensics")

class VisionForensicsAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.biometric_service = BiometricService()
        self.video_service = VideoService()

    @property
    def name(self) -> str:
        return "VisionForensicsAgent"

    async def process(self, case_id: str, context: Dict[str, Any]) -> List[ThreatSignal]:
        signals = []
        sanitized_files = context.get("sanitized_files", [])
        face_embeddings = context.get("face_embeddings_hashes", [])
        
        # Identify selfie images (images that are not video frames, doc crops, spectrograms, or recorded document_files)
        selfies = []
        for file_p in sanitized_files:
            if file_p.lower().endswith((".jpg", ".jpeg", ".png")):
                fn = os.path.basename(file_p).lower()
                is_document = file_p in context.get("document_files", [])
                if "frame" not in fn and "doc_portrait" not in fn and "spectrogram" not in fn and not is_document:
                    selfies.append(file_p)
        logger.info(f"Identified selfies for biometric face verification: {[os.path.basename(s) for s in selfies]}")
        
        files_to_process = list(sanitized_files)
        for file_path in files_to_process:
            is_video = file_path.lower().endswith((".mp4", ".webm"))
            
            if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
                target_file = file_path
            elif is_video:
                logger.info(f"Running video forensics & tracking on video '{os.path.basename(file_path)}'...")
                
                # Run video-specific tracking and liveness checks
                tracking_data = self.video_service.track_faces_in_video(file_path)
                active_liveness = self.video_service.verify_active_liveness(file_path)
                temporal_forensics = self.video_service.verify_video_deepfake_threats(file_path)

                # Add signals if threats detected in video track
                if not active_liveness["challenge_response_success"]:
                    signals.append(ThreatSignal(
                        engine_name=self.name,
                        category=ThreatCategory.FACE_SPOOF,
                        confidence_score=1.0 - active_liveness["active_liveness_score"],
                        severity="HIGH",
                        description=f"Active liveness challenge-response verification failed for video '{os.path.basename(file_path)}'. Expected movement patterns not detected.",
                        evidence_payload={
                            "engine": "MediaPipe-Liveness",
                            "active_liveness_score": active_liveness["active_liveness_score"],
                            "blinks_count": active_liveness["blinks_count"]
                        }
                    ))

                if temporal_forensics["deepfake_score"] > 0.4:
                    signals.append(ThreatSignal(
                        engine_name=self.name,
                        category=ThreatCategory.DEEPFAKE_IMAGE,
                        confidence_score=temporal_forensics["deepfake_score"],
                        severity="CRITICAL" if temporal_forensics["deepfake_score"] > 0.75 else "HIGH",
                        description=(
                            f"Video deepfake forensics flagged '{os.path.basename(file_path)}' as synthetic. "
                            f"Anomalies: {', '.join(temporal_forensics['detected_anomalies'])}."
                        ),
                        evidence_payload={
                            "engine": "TimeSformer-XCLIP",
                            "deepfake_score": temporal_forensics["deepfake_score"],
                            "detected_anomalies": temporal_forensics["detected_anomalies"],
                            "temporal_consistency": temporal_forensics["temporal_consistency"]
                        }
                    ))

                # Extract keyframe for standard image checks
                logger.info(f"Extracting frame from video '{os.path.basename(file_path)}' for visual deepfake diagnostics...")
                target_file = self._extract_video_frame(file_path)
                if not target_file or not os.path.exists(target_file):
                    logger.error(f"Could not extract frame from video '{os.path.basename(file_path)}' for vision forensics.")
                    continue
                # Register the extracted frame as a debug image!
                if target_file not in context["sanitized_files"]:
                    context["sanitized_files"].append(target_file)
                
                # Perform biometric verification matching video face vs selfie
                for selfie_path in selfies:
                    match, sim = self.biometric_service.verify_faces_match(target_file, selfie_path)
                    logger.info(f"Biometric face match between video frame '{os.path.basename(target_file)}' and selfie '{os.path.basename(selfie_path)}': match={match}, similarity={sim:.4f}")
                    if not match:
                        signals.append(ThreatSignal(
                            engine_name=self.name,
                            category=ThreatCategory.IDENTITY_INCONSISTENCY,
                            confidence_score=1.0 - sim,
                            severity="CRITICAL",
                            description=(
                                f"Biometric verification failed: The face in the video '{os.path.basename(file_path)}' "
                                f"does not match the selfie image '{os.path.basename(selfie_path)}' (Similarity: {sim:.2f})."
                            ),
                            evidence_payload={
                                "engine": "InsightFace-ArcFace",
                                "video_file": os.path.basename(file_path),
                                "selfie_file": os.path.basename(selfie_path),
                                "similarity_score": sim,
                                "match_threshold": 0.4
                            }
                        ))
            else:
                continue
                
            filename = os.path.basename(target_file)
            is_doc_crop = "doc_portrait" in filename.lower()

            # Run local biometric face localization & passive liveness verification on image/extracted frame
            detected_faces = self.biometric_service.detect_face(target_file)
            liveness_data = self.biometric_service.verify_passive_liveness(target_file)
            
            # Generate stable ArcFace representation hash
            face_hash = self.biometric_service.get_face_embedding(target_file)
            if face_hash not in face_embeddings:
                face_embeddings.append(face_hash)

            # Generate threat signal if local passive liveness verification failed
            if liveness_data["spoof_detected"]:
                signals.append(ThreatSignal(
                    engine_name=self.name,
                    category=ThreatCategory.FACE_SPOOF,
                    confidence_score=liveness_data["confidence"],
                    severity="HIGH",
                    description=f"Local passive liveness verification failed for '{filename}'. Detected spoof category: {liveness_data['spoof_type']}.",
                    evidence_payload={
                        "engine": "SilentFace-Liveness",
                        "liveness_score": liveness_data["liveness_score"],
                        "spoof_type": liveness_data["spoof_type"]
                    }
                ))

            if is_doc_crop:
                # Do not run VLM deepfake checks on cropped document portraits to avoid redundant calls
                continue

            if not settings.NVIDIA_APIKEY:
                logger.error("NVIDIA API Key is missing in configuration settings.")
                raise ValueError("NVIDIA API Key is required for live visual deepfake forensics. No mock fallback allowed.")

            logger.info(f"Invoking meta/llama-3.2-11b-vision-instruct for visual deepfake diagnostics on '{filename}'...")
            nim_signals, biometric_info = await self._invoke_nvidia_nim_forensics(target_file)
            if biometric_info is None:
                logger.error(f"NVIDIA NIM visual forensics failed for '{filename}'.")
                raise ValueError("NVIDIA NIM Deepfake Forensics API execution failed. No mock fallback allowed.")

            # Update shared context with biometric indicators
            if biometric_info.get("estimated_age"):
                context["biometric_estimated_age"] = biometric_info.get("estimated_age")
            if biometric_info.get("estimated_gender"):
                context["biometric_estimated_gender"] = biometric_info.get("estimated_gender")
            
            signals.extend(nim_signals)
            
        context["face_embeddings_hashes"] = face_embeddings
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
                logger.error(f"Invalid frame count ({frame_count}) for video: {file_path}")
                cap.release()
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

    async def _invoke_nvidia_nim_forensics(self, file_path: str) -> tuple[List[ThreatSignal], Dict[str, Any] | None]:
        """
        Sends the base64 image representation to NVIDIA NIM for visual anomaly analysis.
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
                "You are an expert AI visual forensic analyst. Inspect the portrait region of this document or selfie image. "
                "1) Search for GAN/Diffusion artifacts, smoothed skin borders, distorted eyes, asymmetrical structures, or background warping. "
                "2) Estimate the biometric age and gender. "
                "3) Output your findings in raw JSON format inside ```json ... ``` with keys: "
                "'biometric_findings' (object containing: estimated_age (integer), estimated_gender (string)), "
                "'deepfake_score' (float between 0.0 and 1.0 representing deepfake/GAN probability), "
                "'liveness_score' (float between 0.0 and 1.0 representing natural skin liveness probability where 1.0 is highly natural and 0.0 is a printed/screen spoof), "
                "'visual_anomalies' (list of strings outlining localized visual discrepancies). "
                "Do not output any other text besides the JSON block."
            )

            payload = {
                "model": settings.MODELS["nvidia_nim"]["vlm_model"],
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
                    logger.warning(f"NVIDIA NIM error ({response.status_code}) on attempt {attempt + 1}: {response.text}")
                except httpx.HTTPError as exc:
                    logger.warning(f"HTTP connection/timeout error on attempt {attempt + 1}: {str(exc)}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
            
            if not response or response.status_code != 200:
                logger.error("NVIDIA NIM Deepfake Forensics API execution failed after multiple retry attempts.")
                return [], None

            response_data = response.json()
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content")
            if not content:
                logger.error(f"NVIDIA NIM Deepfake Forensics API returned empty or null content choice. Response payload: {response_data}")
                return [], None
            
            json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            json_text = json_match.group(1) if json_match else content
            
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from NVIDIA NIM response. Attempting markdown parsing recovery...")
                data = self._parse_non_json_forensics(content)
                if not data:
                    data = {}
                logger.info("Successfully handled forensics data recovery from markdown/conversational response!")
            
            biometrics = data.get("biometric_findings", {})
            
            raw_df_score = data.get("deepfake_score", 0.0)
            try:
                deepfake_score = float(raw_df_score)
            except (ValueError, TypeError):
                deepfake_score = 0.0
                
            raw_live_score = data.get("liveness_score", 1.0)
            try:
                liveness_score = float(raw_live_score)
            except (ValueError, TypeError):
                liveness_score = 1.0
                
            if deepfake_score > 1.0:
                deepfake_score = deepfake_score / 100.0
            deepfake_score = max(0.0, min(1.0, deepfake_score))
            
            if liveness_score > 1.0:
                liveness_score = liveness_score / 100.0
            liveness_score = max(0.0, min(1.0, liveness_score))
            
            anomalies = data.get("visual_anomalies", [])

            signals = []
            filename = os.path.basename(file_path)
            
            if deepfake_score > 0.4:
                signals.append(ThreatSignal(
                    engine_name=self.name,
                    category=ThreatCategory.DEEPFAKE_IMAGE,
                    confidence_score=deepfake_score,
                    severity="CRITICAL" if deepfake_score > 0.75 else "HIGH",
                    description=f"NVIDIA NIM visual forensics flagged image '{filename}' as synthetic deepfake. Anomalies: {', '.join(anomalies)}",
                    evidence_payload={
                        "nvidia_model": settings.MODELS["nvidia_nim"]["vlm_model"],
                        "detected_anomalies": anomalies,
                        "vlm_deepfake_score": deepfake_score
                    }
                ))
                
            if liveness_score < 0.6:
                signals.append(ThreatSignal(
                    engine_name=self.name,
                    category=ThreatCategory.FACE_SPOOF,
                    confidence_score=1.0 - liveness_score,
                    severity="HIGH",
                    description=f"NVIDIA NIM liveness verification failed for '{filename}'. The portrait exhibits high probability of photo/screen spoofing.",
                    evidence_payload={
                        "nvidia_model": settings.MODELS["nvidia_nim"]["vlm_model"],
                        "liveness_confidence": liveness_score,
                        "spoof_severity": 1.0 - liveness_score
                    }
                ))

            return signals, biometrics

        except Exception as e:
            logger.error(f"NVIDIA NIM deepfake forensics failed: {str(e)}", exc_info=True)
            return [], None

    def _parse_non_json_forensics(self, text: str) -> Dict[str, Any]:
        data = {}
        biometric_findings = {}
        age_match = re.search(r"(?:Estimated Age|Age)[:\s\*-]+(\d+)", text, re.IGNORECASE)
        if age_match:
            try:
                biometric_findings["estimated_age"] = int(age_match.group(1))
            except ValueError:
                pass
                
        gender_match = re.search(r"(?:Estimated Gender|Gender)[:\s\*-]+(Male|Female|Other|Unspecified)", text, re.IGNORECASE)
        if gender_match:
            biometric_findings["estimated_gender"] = gender_match.group(1).strip().capitalize()
        
        data["biometric_findings"] = biometric_findings
        
        df_match = re.search(r"Deepfake Score[:\s\*-]+([0-9.]+)", text, re.IGNORECASE)
        if df_match:
            try:
                data["deepfake_score"] = float(df_match.group(1))
            except ValueError:
                pass
                
        live_match = re.search(r"Liveness Score[:\s\*-]+([0-9.]+)", text, re.IGNORECASE)
        if live_match:
            try:
                data["liveness_score"] = float(live_match.group(1))
            except ValueError:
                pass
                
        anomalies = []
        anomalies_match = re.search(r"Visual Anomalies[:\s\*-]+(.*)", text, re.DOTALL | re.IGNORECASE)
        if anomalies_match:
            anom_block = anomalies_match.group(1).strip()
            for line in anom_block.splitlines():
                cleaned = line.strip().rstrip(".").lower()
                if not cleaned or cleaned in ("none", "none detected", "no anomalies detected"):
                    continue
                anomalies.append(line)
        data["visual_anomalies"] = anomalies
        
        return data
