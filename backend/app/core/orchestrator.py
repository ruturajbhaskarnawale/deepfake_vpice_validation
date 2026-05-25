import os
import logging
from typing import List, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from backend.app.core.config import settings

# Database models
from backend.app.models.db_models import Case, ThreatSignalDB, RiskEvaluationDB, AuditLogDB
from backend.app.models.pydantic_models import JobStatus, AuditAction

# Agents
from backend.app.agents.validator import ValidatorAgent
from backend.app.agents.document_ocr import DocumentOCRAgent
from backend.app.agents.vision_forensics import VisionForensicsAgent
from backend.app.agents.voice_auth import VoiceAuthenticityAgent
from backend.app.agents.identity_graph import IdentityGraphAgent
from backend.app.agents.risk_scorer import RiskScorerAgent

logger = logging.getLogger("sentinel.orchestrator")

class CentralOrchestrator:
    def __init__(self):
        # Initialize detection agents
        self.validator_agent = ValidatorAgent()
        self.ocr_agent = DocumentOCRAgent()
        self.vision_agent = VisionForensicsAgent()
        self.voice_agent = VoiceAuthenticityAgent()
        self.identity_agent = IdentityGraphAgent()
        self.risk_scorer = RiskScorerAgent()
        from backend.app.services.biometric_consistency import BiometricConsistencyEngine
        self.biometric_consistency_engine = BiometricConsistencyEngine()

    async def execute_pipeline(self, db: AsyncSession, case_id: UUID) -> Case:
        """
        Coordinates the multi-stage, multi-modal trust intelligence extraction, 
        evaluating file authenticity, biometric deepfakes, layout anomalies, 
        and performing cross-modal demographic and graph correlation checks.
        """
        logger.info(f"Triggering orchestration pipeline for Case {case_id}")
        
        # 1. Fetch case from Database
        result = await db.execute(select(Case).where(Case.id == case_id))
        case = result.scalar_one_or_none()
        if not case:
            raise ValueError(f"Case {case_id} not found in database")

        # Set status to RUNNING
        case.status = JobStatus.RUNNING
        await db.commit()
        await db.refresh(case)

        # Audit Log: Ingested & Started
        await self._log_audit(db, case_id, AuditAction.INGESTED, "CentralOrchestrator", "Multi-modal orchestration pipeline started.")

        try:
            # Context dictionary passed down between agent nodes
            context: Dict[str, Any] = {
                "files_received": case.files_received,
                "sanitized_files": [],
                "ocr_payload": {},
                "db": db
            }

            # --- STAGE 1: File Ingestion & Validation ---
            sanitized_paths = []
            for file_path in case.files_received:
                is_valid, mime, err = self.validator_agent.validate_file(file_path)
                if not is_valid:
                    # Update state to FAILED
                    case.status = JobStatus.FAILED
                    await db.commit()
                    await self._log_audit(db, case_id, AuditAction.VALIDATED, "ValidatorAgent", f"File validation failed: {err}")
                    raise ValueError(f"File validation failed: {err}")
                
                # Sanitize media (normalize orientation, dimensions, frequencies)
                sanitized_dir = os.path.join(os.path.dirname(file_path), "sanitized")
                sanitized_file = self.validator_agent.sanitize_media(file_path, sanitized_dir)
                sanitized_paths.append(sanitized_file)

            context["sanitized_files"] = sanitized_paths
            case.sanitized_files = sanitized_paths
            await db.commit()
            
            await self._log_audit(db, case_id, AuditAction.VALIDATED, "ValidatorAgent", "All media assets successfully validated and sanitized.")

            # --- STAGE 2: Multi-Modal Detection Engines ---
            all_detected_signals = []

            # 2.1 Document OCR / Layout Tampering
            ocr_signals = await self.ocr_agent.process(str(case_id), context)
            all_detected_signals.extend(ocr_signals)

            # 2.2 Image Forensics / Face Verification
            vision_signals = await self.vision_agent.process(str(case_id), context)
            all_detected_signals.extend(vision_signals)

            voice_signals = await self.voice_agent.process(str(case_id), context)
            all_detected_signals.extend(voice_signals)

            # 2.4 Biometric Continuity / Consistency Verification
            selfie_file = None
            video_file = None
            for file_p in context.get("sanitized_files", []):
                fn = os.path.basename(file_p).lower()
                if file_p.lower().endswith((".jpg", ".jpeg", ".png")):
                    if "frame" not in fn and "doc_portrait" not in fn and "spectrogram" not in fn:
                        selfie_file = file_p
                elif file_p.lower().endswith((".mp4", ".webm")):
                    video_file = file_p

            if selfie_file and video_file:
                logger.info(f"Running biometric consistency engine on selfie: {os.path.basename(selfie_file)}, video: {os.path.basename(video_file)}")
                consistency_signals, continuity_metadata = self.biometric_consistency_engine.process_continuity(selfie_file, video_file)
                all_detected_signals.extend(consistency_signals)
                context["face_match_confidence"] = continuity_metadata["face_match_confidence"]
                context["frame_stability_score"] = continuity_metadata["frame_stability_score"]
                context["biometric_continuity_metadata"] = continuity_metadata
            else:
                context["face_match_confidence"] = 1.0
                context["frame_stability_score"] = 1.0
                context["biometric_continuity_metadata"] = {}

            # --- STAGE 3: Synthetic Identity Verification ---
            # Evaluates demographic mismatches (stated details vs biometric results) 
            # and triggers Neo4j similarity matches.
            identity_signals = await self.identity_agent.process(str(case_id), context)
            all_detected_signals.extend(identity_signals)

            # Save detected threat signals in Database
            for sig in all_detected_signals:
                db_sig = ThreatSignalDB(
                    id=sig.id,
                    case_id=case_id,
                    engine_name=sig.engine_name,
                    category=sig.category,
                    confidence_score=sig.confidence_score,
                    severity=sig.severity,
                    description=sig.description,
                    evidence_payload=sig.evidence_payload,
                    timestamp=sig.timestamp
                )
                db.add(db_sig)
            
            await db.commit()

            # --- STAGE 4: Risk Scoring & Classification ---
            # Correlate weak/strong patterns and calculate unified Risk Evaluation
            risk_eval = await self.risk_scorer.evaluate_risk(case_id, all_detected_signals, db=db, context=context)

            db_eval = RiskEvaluationDB(
                id=risk_eval.id,
                case_id=case_id,
                composite_risk_score=risk_eval.composite_risk_score,
                risk_level=risk_eval.risk_level,
                triggered_signals_count=risk_eval.triggered_signals_count,
                signals_summary=risk_eval.signals_summary,
                recommendation=risk_eval.recommendation,
                evaluated_at=risk_eval.evaluated_at
            )
            db.add(db_eval)

            # --- STAGE 5: Complete & Close Workflow ---
            ocr_payload = context.get("ocr_payload", {})
            ocr_payload["face_embeddings_hashes"] = context.get("face_embeddings_hashes", [])
            ocr_payload["voice_embeddings_hashes"] = context.get("voice_embeddings_hashes", [])
            ocr_payload["biometric_estimated_age"] = context.get("biometric_estimated_age")
            ocr_payload["biometric_estimated_gender"] = context.get("biometric_estimated_gender")
            ocr_payload["voice_transcript"] = context.get("voice_transcript")
            ocr_payload["voice_demographics"] = context.get("voice_demographics")
            case.ocr_payload = ocr_payload
            case.sanitized_files = context.get("sanitized_files", [])
            case.debug_images = context.get("debug_images", [])
            case.status = JobStatus.COMPLETED
            await db.commit()

            await self._log_audit(
                db, case_id, AuditAction.ANALYSIS_COMPLETED, "CentralOrchestrator", 
                f"Analysis finalized. Level: {risk_eval.risk_level.value}. Recommendation: {risk_eval.recommendation}."
            )

            await db.refresh(case)
            return case

        except Exception as e:
            logger.error(f"Pipeline processing failed for case {case_id}: {str(e)}", exc_info=True)
            case.status = JobStatus.FAILED
            await db.commit()
            await self._log_audit(db, case_id, AuditAction.SYSTEM_DECISION, "CentralOrchestrator", f"Pipeline aborted with exception: {str(e)}")
            raise e

    async def _log_audit(self, db: AsyncSession, case_id: UUID, action: AuditAction, actor: str, details: str):
        audit = AuditLogDB(
            case_id=case_id,
            action=action,
            actor=actor,
            details=details
        )
        db.add(audit)
        await db.commit()
