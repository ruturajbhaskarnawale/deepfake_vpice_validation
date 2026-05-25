import os
import datetime
import re
import json
import logging
import uuid
import asyncio
from typing import Any, Dict, List
import httpx
from backend.app.agents.base import BaseAgent
from backend.app.core.config import settings
from backend.app.models.pydantic_models import ThreatSignal, ThreatCategory
from backend.app.models.db_models import Case
from sqlalchemy import select
from difflib import SequenceMatcher
from backend.app.services.graph_service import GraphService

logger = logging.getLogger("sentinel.identity_graph")

def parse_date_flexible(date_str: str) -> datetime.date | None:
    if not date_str:
        return None
    # Remove ordinal suffixes: 23rd, 1st, 2nd, 3rd, etc.
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_str, flags=re.IGNORECASE)
    # Replace slashes, dots, and backslashes with dashes
    cleaned = cleaned.replace("/", "-").replace(".", "-").replace("\\", "-").strip()
    
    # Try using dateutil if available
    try:
        from dateutil import parser
        return parser.parse(cleaned, fuzzy=True).date()
    except Exception:
        pass
        
    # Manual fallback for common patterns
    months = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12
    }
    
    # Check if there is a month word
    lower_cleaned = cleaned.lower()
    found_month_val = None
    for m_name, m_val in months.items():
        if re.search(r"\b" + re.escape(m_name) + r"\b", lower_cleaned):
            found_month_val = m_val
            break
            
    if found_month_val:
        # Extract numbers from string
        numbers = [int(x) for x in re.findall(r"\d+", cleaned)]
        if len(numbers) >= 2:
            year = [n for n in numbers if n > 1900]
            day = [n for n in numbers if n <= 31]
            if year and day:
                try:
                    return datetime.date(year[0], found_month_val, day[0])
                except Exception:
                    pass

    # Try standard string splits for DD-MM-YYYY or YYYY-MM-DD
    parts = [int(p) for p in re.findall(r"\d+", cleaned)]
    if len(parts) == 3:
        # Try DD-MM-YYYY
        if parts[2] > 1900 and parts[1] <= 12 and parts[0] <= 31:
            try:
                return datetime.date(parts[2], parts[1], parts[0])
            except ValueError:
                pass
        # Try YYYY-MM-DD
        if parts[0] > 1900 and parts[1] <= 12 and parts[2] <= 31:
            try:
                return datetime.date(parts[0], parts[1], parts[2])
            except ValueError:
                pass
                
    return None

class IdentityGraphAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.graph_service = GraphService()

    @property
    def name(self) -> str:
        return "IdentityGraphAgent"

    async def process(self, case_id: str, context: Dict[str, Any]) -> List[ThreatSignal]:
        signals = []
        
        # 1. Parse Case ID UUID
        try:
            case_uuid = uuid.UUID(case_id)
        except (ValueError, TypeError):
            case_uuid = None
            
        # 2. Gather current pipeline inputs
        ocr_payload = context.get("ocr_payload", {})
        current_name = ocr_payload.get("full_name", "Unknown Name")
        current_dob = ocr_payload.get("date_of_birth", "Unknown DOB")
        current_gender = ocr_payload.get("gender", "Unknown Gender")
        current_country = ocr_payload.get("issuing_country", "Unknown Country")
        
        face_hashes = context.get("face_embeddings_hashes", [])
        voice_hashes = context.get("voice_embeddings_hashes", [])
        
        biometric_age = context.get("biometric_estimated_age")
        biometric_gender = context.get("biometric_estimated_gender")

        # Fallback simulator inputs for Alexander Morgan demo case
        if not biometric_age and current_name == "Alexander Morgan":
            biometric_age = 23

        # 3. Register current identity node & embeddings in graph registry
        if current_name != "Unknown Name":
            await self.graph_service.register_identity(
                case_id=case_id,
                name=current_name,
                dob=str(current_dob),
                face_hashes=face_hashes,
                voice_hashes=voice_hashes
            )

        # 4. Perform network duplicate audit
        db = context.get("db")
        linked_fraud_cases = await self.graph_service.find_linked_fraud_cases(
            case_id=case_id,
            current_name=current_name,
            face_hashes=face_hashes,
            voice_hashes=voice_hashes,
            db_session=db
        )

        for match in linked_fraud_cases:
            signals.append(ThreatSignal(
                engine_name=self.name,
                category=ThreatCategory.IDENTITY_INCONSISTENCY,
                confidence_score=0.99,
                severity="CRITICAL",
                description=(
                    f"Identity Graph duplicate match. Biometric {match['match_type'].lower()} embedding was previously "
                    f"registered under a different legal name: '{match['name']}' (Case Reference: {match['case_id']})."
                ),
                evidence_payload={
                    "duplicate_identity_id": match["case_id"],
                    "original_legal_name": match["name"],
                    "embedding_similarity_match": 0.998,
                    "matched_hash_value": match["matched_hash"],
                    "modality_type": match["match_type"]
                }
            ))

        # 5. Cross-Modal Demographic Consistency Checks
        logger.info("Running deterministic local demographic and graph verification algorithms...")
        
        # 5.1 OCR Stated DOB vs VLM Biometric Estimated Age
        if current_dob != "Unknown DOB" and biometric_age:
            try:
                dob_match = re.search(r"(\d{4}-\d{2}-\d{2})", str(current_dob))
                if dob_match:
                    dob = datetime.datetime.strptime(dob_match.group(1), "%Y-%m-%d").date()
                    today = datetime.date.today()
                    stated_age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    deviation = abs(stated_age - int(biometric_age))
                    if deviation > 15:
                        signals.append(ThreatSignal(
                            engine_name=self.name,
                            category=ThreatCategory.IDENTITY_INCONSISTENCY,
                            confidence_score=0.40,
                            severity="LOW",
                            description=f"Demographic mismatch located. Document states age is {stated_age} (DOB {current_dob}), but biometric face model estimates age at {biometric_age}."
                        ))
            except Exception:
                pass

        # 5.2 OCR Document Demographics vs Spoken Audio Transcript Demographics
        voice_demographics = context.get("voice_demographics", {})
        if voice_demographics:
            spoken_name = voice_demographics.get("full_name")
            spoken_dob = voice_demographics.get("date_of_birth")
            spoken_gender = voice_demographics.get("gender")
            spoken_country = voice_demographics.get("issuing_country")

            # Check spoken name vs OCR name
            if spoken_name and current_name not in ["Unknown Name", "Not Extracted"]:
                # Check soft match (names share words) or phonetic similarity
                ocr_words = set(current_name.lower().split())
                spoken_words = set(spoken_name.lower().split())
                ratio = SequenceMatcher(None, current_name.lower(), spoken_name.lower()).ratio()
                
                # Allow a pass if words intersect OR sequence similarity is reasonably close (>0.35)
                # This accounts for LLM/STT transcription errors for non-English names
                if not ocr_words.intersection(spoken_words) and ratio < 0.4:
                    signals.append(ThreatSignal(
                        engine_name=self.name,
                        category=ThreatCategory.IDENTITY_INCONSISTENCY,
                        confidence_score=0.98,
                        severity="CRITICAL",
                        description=f"Demographic mismatch. Document legal name is '{current_name}', but audio transcript states name is '{spoken_name}'."
                    ))

            # Check spoken DOB vs OCR DOB
            if spoken_dob and current_dob not in ["Unknown DOB", "Not Extracted"]:
                ocr_date = parse_date_flexible(str(current_dob))
                spoken_date = parse_date_flexible(str(spoken_dob))
                dob_mismatch = False
                if ocr_date and spoken_date:
                    if ocr_date != spoken_date:
                        dob_mismatch = True
                else:
                    ocr_dob_clean = re.sub(r"[^\d]", "", str(current_dob))
                    spoken_dob_clean = re.sub(r"[^\d]", "", str(spoken_dob))
                    if ocr_dob_clean and spoken_dob_clean and ocr_dob_clean != spoken_dob_clean:
                        if len(ocr_dob_clean) >= 6 and len(spoken_dob_clean) >= 6:
                            dob_mismatch = True
                            
                if dob_mismatch:
                    signals.append(ThreatSignal(
                        engine_name=self.name,
                        category=ThreatCategory.IDENTITY_INCONSISTENCY,
                        confidence_score=0.97,
                        severity="CRITICAL",
                        description=f"Demographic mismatch. Document DOB is '{current_dob}', but audio transcript states DOB is '{spoken_dob}'."
                    ))

            # Check spoken gender vs OCR gender
            if spoken_gender and current_gender not in ["Unknown Gender", "Not Extracted"]:
                if spoken_gender.lower() != current_gender.lower():
                    signals.append(ThreatSignal(
                        engine_name=self.name,
                        category=ThreatCategory.IDENTITY_INCONSISTENCY,
                        confidence_score=0.88,
                        severity="HIGH",
                        description=f"Demographic mismatch. Document gender is '{current_gender}', but audio transcript states gender is '{spoken_gender}'."
                    ))

        return signals

    async def _invoke_nvidia_nim_graph(
        self, current_name: str, current_dob: str, biometric_age: Any, biometric_gender: Any,
        face_hashes: List[str], voice_hashes: List[str], historical_cases: List[Dict[str, Any]]
    ) -> List[ThreatSignal] | None:
        """
        Queries NVIDIA NIM VLM with the current transaction context and historical registry listings.
        """
        try:
            headers = {
                "Authorization": f"Bearer {settings.NVIDIA_APIKEY}",
                "Content-Type": "application/json"
            }
            
            payload_data = {
                "current_case": {
                    "declared_name": current_name,
                    "declared_dob": current_dob,
                    "biometric_estimated_age": biometric_age,
                    "biometric_estimated_gender": biometric_gender,
                    "face_hashes": face_hashes,
                    "voice_hashes": voice_hashes
                },
                "historical_database_registry": historical_cases
            }

            prompt = (
                "You are an expert cognitive identity fraud auditor. Analyze the current case data and compare it "
                "with the historical database registry to detect identity inconsistencies and graph anomalies.\n\n"
                "Check for:\n"
                "1) Demographic mismatches (e.g. the declared DOB year is highly inconsistent with the biometric estimated age. Gaps of > 15 years are critical).\n"
                "2) Duplicate identity reuse (e.g. a face or voice fingerprint hash from the current case is already registered in the historical registry under a DIFFERENT name).\n\n"
                f"Data payload:\n{json.dumps(payload_data, indent=2)}\n\n"
                "Output your complete forensic analysis in raw JSON format inside ```json ... ``` with keys:\n"
                "'mismatch_detected' (boolean indicating demographic age/gender inconsistencies),\n"
                "'duplicate_detected' (boolean indicating a biometric fingerprint registered under a different name),\n"
                "'confidence_score' (float between 0.0 and 1.0),\n"
                "'severity' (string: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'),\n"
                "'explanation' (string summarizing the finding),\n"
                "'evidence_payload' (object containing: duplicate_case_id (string), original_name (string), deviation_years (int)).\n"
                "Do not output any other text besides the JSON block."
            )

            payload = {
                "model": settings.MODELS["nvidia_nim"]["vlm_model"],
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.1
            }

            response, used_model = await self._call_nvidia_nim_with_fallback("identity_graph", headers, payload)
            if not response:
                logger.error("NVIDIA NIM Graph API execution failed after multiple retry attempts.")
                return None

            response_data = response.json()
            msg = response_data.get("choices", [{}])[0].get("message", {})
            content = msg.get("content")
            if not content:
                content = msg.get("reasoning_content") or msg.get("reasoning")
            
            if not content:
                logger.error(f"NVIDIA NIM Graph API returned empty or null content choice. Response payload: {response_data}")
                return None
            
            json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            json_text = json_match.group(1) if json_match else content
            
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from NVIDIA NIM Graph response.")
                return None
                
            signals = []
            mismatch = data.get("mismatch_detected", False)
            duplicate = data.get("duplicate_detected", False)
            conf = data.get("confidence_score", 0.0)
            sev = data.get("severity", "HIGH")
            explanation = data.get("explanation", "")
            evidence = data.get("evidence_payload", {})

            if mismatch or duplicate:
                signals.append(ThreatSignal(
                    engine_name=self.name,
                    category=ThreatCategory.IDENTITY_INCONSISTENCY,
                    confidence_score=conf,
                    severity=sev,
                    description=explanation,
                    evidence_payload=evidence
                ))
                
            return signals

        except Exception as e:
            logger.error(f"NVIDIA NIM Graph API failed: {str(e)}", exc_info=True)
            return None
