import logging
import json
import asyncio
import datetime
from typing import Any, Dict, List, Optional
import httpx
from backend.app.agents.base import BaseAgent
from backend.app.core.config import settings
from backend.app.models.pydantic_models import RiskEvaluation, RiskLevel, ThreatSignal, ThreatCategory

logger = logging.getLogger("sentinel.risk_scorer")

class RiskScorerAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "RiskScorerAgent"

    async def process(self, case_id: str, context: Dict[str, Any]) -> List[ThreatSignal]:
        # The risk scorer outputs a RiskEvaluation object, which is appended to the context.
        # It does not create individual threat signals.
        return []

    async def evaluate_risk(
        self, case_id: Any, signals: List[ThreatSignal], db: Optional[Any] = None, context: Optional[Dict[str, Any]] = None
    ) -> RiskEvaluation:
        """
        Compiles individual threat signals, applies dynamic multi-layer risk layering,
        signal correlation rules, velocity checks, historical fraud list memory,
        trust score calculation, and adaptive thresholds to return an enriched RiskEvaluation.
        """
        if context is None:
            context = {}

        # 1. VELOCITY ANALYSIS
        velocity_exceeded = False
        rapid_retries = False
        prior_attempts_count = 0
        prior_rejections_count = 0
        biometric_reuse_detected = False
        historical_risk_modifier = 0.0

        if db is not None:
            try:
                from backend.app.models.db_models import Case
                from sqlalchemy import select
                stmt = select(Case).order_by(Case.created_at.desc())
                res = await db.execute(stmt)
                historical_cases = res.scalars().all()

                prior_attempts_count = len(historical_cases) - 1 # exclude current case
                
                # Check velocity (attempts in last hour)
                one_hour_ago = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
                attempts_last_hour = 0
                for c in historical_cases:
                    # c.created_at might be offset-naive or offset-aware, strip tz if needed
                    c_time = c.created_at.replace(tzinfo=None) if c.created_at.tzinfo else c.created_at
                    if c_time >= one_hour_ago:
                        attempts_last_hour += 1

                # If velocity is high, generate signals and flag
                if attempts_last_hour > 5:
                    velocity_exceeded = True
                    signals.append(ThreatSignal(
                        engine_name="RiskScorerAgent",
                        category=ThreatCategory.VELOCITY_EXCEEDED,
                        confidence_score=0.95,
                        severity="HIGH",
                        description=f"Velocity threshold exceeded: {attempts_last_hour} verification attempts in the last hour."
                    ))

                if attempts_last_hour > 3:
                    rapid_retries = True
                    signals.append(ThreatSignal(
                        engine_name="RiskScorerAgent",
                        category=ThreatCategory.RAPID_RETRY_PATTERN,
                        confidence_score=0.85,
                        severity="MEDIUM",
                        description="Rapid retry pattern detected. Multiple sequential upload attempts observed."
                    ))

                # Check biometric reuse and rejection logs
                current_face_hashes = context.get("face_embeddings_hashes", [])
                current_voice_hashes = context.get("voice_embeddings_hashes", [])

                for c in historical_cases:
                    if str(c.id) == str(case_id):
                        continue
                    
                    # Track rejections
                    is_rejected = False
                    if c.risk_evaluation and c.risk_evaluation.recommendation == "REJECT":
                        prior_rejections_count += 1
                        is_rejected = True

                    # Match biometric overlap
                    c_payload = c.ocr_payload or {}
                    c_faces = c_payload.get("face_embeddings_hashes", [])
                    c_voices = c_payload.get("voice_embeddings_hashes", [])

                    match_found = False
                    for fh in current_face_hashes:
                        if fh in c_faces:
                            match_found = True
                            break
                    for vh in current_voice_hashes:
                        if vh in c_voices:
                            match_found = True
                            break

                    if match_found:
                        biometric_reuse_detected = True
                        if is_rejected:
                            # Higher risk modifier if previously rejected
                            historical_risk_modifier += 30.0
                        else:
                            historical_risk_modifier += 15.0

                if biometric_reuse_detected:
                    signals.append(ThreatSignal(
                        engine_name="RiskScorerAgent",
                        category=ThreatCategory.IDENTITY_INCONSISTENCY,
                        confidence_score=0.99,
                        severity="CRITICAL",
                        description="Historical biometric reuse detected. Embeddings match historical cases."
                    ))

            except Exception as e:
                logger.warning(f"Failed to query historical fraud memory: {str(e)}")

        if not signals:
            # Benign case fallback
            return RiskEvaluation(
                case_id=case_id,
                composite_risk_score=5.0,
                risk_level=RiskLevel.LOW,
                triggered_signals_count=0,
                signals_summary=[{
                    "category": "SYSTEM_SUMMARY",
                    "severity": "LOW",
                    "confidence": 1.0,
                    "description": "All multi-modal verification checks completed successfully. No threat vectors detected.",
                    "fraud_payload": {
                        "risk_layers": {
                            "document_risk": 0.0,
                            "biometric_risk": 0.0,
                            "consistency_risk": 0.0,
                            "network_graph_risk": 0.0,
                            "behavioral_risk": 0.0
                        },
                        "correlations": [],
                        "historical_context": {
                            "prior_attempts": prior_attempts_count,
                            "prior_rejections": prior_rejections_count,
                            "biometric_reuse_detected": biometric_reuse_detected,
                            "historical_risk_modifier": historical_risk_modifier
                        },
                        "trust_metrics": {
                            "trust_score": 100.0,
                            "trust_factors": ["No threat signals present", "Prinstine biometric matching verification"]
                        },
                        "threshold_policy": {
                            "base_threshold": 45.0,
                            "dynamic_threshold": 45.0,
                            "reasoning": "Baseline standard check"
                        },
                        "final_decision_reasoning": ["Baseline approve"]
                    }
                }],
                recommendation="APPROVE"
            )

        # 2. HIERARCHICAL RISK LAYERING
        document_risk = 0.0
        biometric_risk = 0.0
        consistency_risk = 0.0
        network_graph_risk = 0.0
        behavioral_risk = 0.0

        signals_summary_list = []
        semantic_keys = set()

        for signal in signals:
            sev = signal.severity.upper()
            weight = 5.0
            if sev == "CRITICAL":
                weight = 45.0
            elif sev == "HIGH":
                weight = 30.0
            elif sev == "MEDIUM":
                weight = 15.0
            
            weighted_contribution = weight * signal.confidence_score

            # Map signal to category layer
            cat = signal.category
            desc = signal.description.lower()
            
            # Map semantic keys for correlation rules
            if cat == ThreatCategory.DOCUMENT_TAMPERING:
                document_risk += weighted_contribution
                semantic_keys.add("OCR_TAMPER")
            elif cat == ThreatCategory.METADATA_ANOMALY:
                document_risk += weighted_contribution
                semantic_keys.add("OCR_TAMPER")
            elif cat in (ThreatCategory.FACE_SPOOF, ThreatCategory.DEEPFAKE_IMAGE):
                biometric_risk += weighted_contribution
                semantic_keys.add("FACE_SPOOF")
            elif cat == ThreatCategory.SYNTHETIC_VOICE:
                biometric_risk += weighted_contribution
                semantic_keys.add("VOICE_CLONE")
            elif cat == ThreatCategory.IDENTITY_INCONSISTENCY:
                if "demographic" in desc or "dob" in desc or "name" in desc:
                    consistency_risk += weighted_contribution
                    semantic_keys.add("DOB_MISMATCH")
                elif "duplicate" in desc or "reuse" in desc:
                    network_graph_risk += weighted_contribution
                    semantic_keys.add("FACE_REUSE")
                elif "biometric" in desc or "match" in desc:
                    consistency_risk += weighted_contribution
                    semantic_keys.add("FACE_IDENTITY_MISMATCH")
                else:
                    consistency_risk += weighted_contribution
            elif cat == ThreatCategory.VELOCITY_EXCEEDED:
                behavioral_risk += weighted_contribution
                semantic_keys.add("MULTI_ACCOUNT")
            elif cat == ThreatCategory.RAPID_RETRY_PATTERN:
                behavioral_risk += weighted_contribution

            signals_summary_list.append({
                "category": signal.category.value,
                "severity": signal.severity,
                "confidence": signal.confidence_score,
                "description": signal.description
            })

        # Cap subscores
        document_risk = min(document_risk, 100.0)
        biometric_risk = min(biometric_risk, 100.0)
        consistency_risk = min(consistency_risk, 100.0)
        network_graph_risk = min(network_graph_risk, 100.0)
        behavioral_risk = min(behavioral_risk, 100.0)

        # 3. SIGNAL CORRELATION INTELLIGENCE ENGINE
        # Rules:
        # ("FACE_SPOOF", "VOICE_CLONE"): 35
        # ("OCR_TAMPER", "DOB_MISMATCH"): 20
        # ("FACE_REUSE", "MULTI_ACCOUNT"): 45
        # ("FACE_IDENTITY_MISMATCH", "VOICE_CLONE"): 40
        CORRELATION_RULES = {
            ("FACE_SPOOF", "VOICE_CLONE"): (35, "Coordinated synthetic identity/avatar indicators detected"),
            ("OCR_TAMPER", "DOB_MISMATCH"): (20, "Coordinated document forgery and demographic inconsistencies detected"),
            ("FACE_REUSE", "MULTI_ACCOUNT"): (45, "Identity farming attempt utilizing reused biometric signatures"),
            ("FACE_IDENTITY_MISMATCH", "VOICE_CLONE"): (40, "Attempted presentation spoofing using cloned audio track")
        }

        correlation_hits = []
        correlation_bonus_total = 0.0

        for pair, (base_bonus, reason) in CORRELATION_RULES.items():
            if pair[0] in semantic_keys and pair[1] in semantic_keys:
                # Confidence-aware correlation bonus
                conf_a = 1.0
                conf_b = 1.0
                
                # Fetch confidence score from matched signals
                for s in signals:
                    s_cat = s.category
                    s_desc = s.description.lower()
                    
                    # Map back categories
                    if pair[0] == "OCR_TAMPER" and s_cat in (ThreatCategory.DOCUMENT_TAMPERING, ThreatCategory.METADATA_ANOMALY):
                        conf_a = s.confidence_score
                    elif pair[0] == "FACE_SPOOF" and s_cat in (ThreatCategory.FACE_SPOOF, ThreatCategory.DEEPFAKE_IMAGE):
                        conf_a = s.confidence_score
                    elif pair[0] == "VOICE_CLONE" and s_cat == ThreatCategory.SYNTHETIC_VOICE:
                        conf_a = s.confidence_score
                    elif pair[0] == "FACE_IDENTITY_MISMATCH" and s_cat == ThreatCategory.IDENTITY_INCONSISTENCY and ("biometric" in s_desc or "match" in s_desc):
                        conf_a = s.confidence_score
                    elif pair[0] == "FACE_REUSE" and s_cat == ThreatCategory.IDENTITY_INCONSISTENCY and ("duplicate" in s_desc or "reuse" in s_desc):
                        conf_a = s.confidence_score
                        
                    if pair[1] == "VOICE_CLONE" and s_cat == ThreatCategory.SYNTHETIC_VOICE:
                        conf_b = s.confidence_score
                    elif pair[1] == "DOB_MISMATCH" and s_cat == ThreatCategory.IDENTITY_INCONSISTENCY and ("demographic" in s_desc or "dob" in s_desc or "name" in s_desc):
                        conf_b = s.confidence_score
                    elif pair[1] == "MULTI_ACCOUNT" and s_cat == ThreatCategory.VELOCITY_EXCEEDED:
                        conf_b = s.confidence_score

                avg_conf = (conf_a + conf_b) / 2.0
                bonus = base_bonus * avg_conf
                correlation_bonus_total += bonus
                correlation_hits.append({
                    "signals": list(pair),
                    "bonus": round(bonus, 2),
                    "reason": reason
                })

        # Calculate base composite score
        total_signals_weight = 0.0
        for signal in signals:
            sev = signal.severity.upper()
            weight = 5.0
            if sev == "CRITICAL":
                weight = 45.0
            elif sev == "HIGH":
                weight = 30.0
            elif sev == "MEDIUM":
                weight = 15.0
            total_signals_weight += weight * signal.confidence_score

        composite_score = min(total_signals_weight + correlation_bonus_total + historical_risk_modifier, 100.0)

        # Enforce threshold overrides for CRITICAL signals
        critical_signals = [s for s in signals if s.severity.upper() == "CRITICAL"]
        if critical_signals:
            max_critical_conf = max(s.confidence_score for s in critical_signals)
            if max_critical_conf >= 0.7:
                composite_score = max(composite_score, 75.0)
            else:
                composite_score = max(composite_score, 50.0)

        # 4. TRUST SCORE ENGINE
        trust_score = 100.0
        trust_factors = []
        
        # Deduct trust based on threat signals presence
        for signal in signals:
            sev = signal.severity.upper()
            deduction = 5.0
            if sev == "CRITICAL":
                deduction = 35.0
            elif sev == "HIGH":
                deduction = 25.0
            elif sev == "MEDIUM":
                deduction = 15.0
            trust_score -= deduction * signal.confidence_score

        # Check for face consistency matching (extracted from context if present)
        face_match_confidence = context.get("face_match_confidence", 1.0)
        if face_match_confidence < 0.70:
            trust_score -= 30.0 * (1.0 - face_match_confidence)
            trust_factors.append("Low facial consistency between selfie and video recording")
        elif face_match_confidence >= 0.85:
            trust_factors.append("High face embedding consistency verified")

        if not any(s.category == ThreatCategory.SYNTHETIC_VOICE for s in signals):
            trust_factors.append("Stable voice biometric profile verified")
        else:
            trust_factors.append("Acoustic deepfake indicators present")

        if not any(s.category in (ThreatCategory.DOCUMENT_TAMPERING, ThreatCategory.METADATA_ANOMALY) for s in signals):
            trust_factors.append("High OCR document layout structure trust")
        else:
            trust_factors.append("Document image layout anomalies detected")

        trust_score = max(0.0, min(100.0, trust_score))

        # 5. ADAPTIVE THRESHOLD ENGINE
        base_threshold = 45.0
        threshold_reasons = ["Baseline standard threshold"]

        if velocity_exceeded or rapid_retries:
            base_threshold -= 10.0
            threshold_reasons.append("Stricter policy applied: velocity limits exceeded")

        if biometric_reuse_detected:
            base_threshold -= 15.0
            threshold_reasons.append("Stricter policy applied: historical biometric duplication linked")

        if trust_score >= 80.0:
            base_threshold += 10.0
            threshold_reasons.append("Lighter policy applied: high session trust verified")

        dynamic_rejection_threshold = base_threshold
        dynamic_escalation_threshold = max(20.0, base_threshold - 25.0)

        # Re-evaluate final decisions based on adaptive thresholds
        if composite_score >= dynamic_rejection_threshold:
            risk_level = RiskLevel.CRITICAL
            recommendation = "REJECT"
        elif composite_score >= dynamic_escalation_threshold:
            risk_level = RiskLevel.HIGH
            recommendation = "ESCALATE_TO_HUMAN"
        else:
            risk_level = RiskLevel.LOW
            recommendation = "APPROVE"

        # Generate LLM forensices correlation summary
        explanation = await self._generate_ai_risk_summary(signals, composite_score)

        # 6. ENRICHED FORENSIC PAYLOAD
        fraud_payload = {
            "risk_layers": {
                "document_risk": round(document_risk, 2),
                "biometric_risk": round(biometric_risk, 2),
                "consistency_risk": round(consistency_risk, 2),
                "network_graph_risk": round(network_graph_risk, 2),
                "behavioral_risk": round(behavioral_risk, 2)
            },
            "correlations": correlation_hits,
            "historical_context": {
                "prior_attempts": prior_attempts_count,
                "prior_rejections": prior_rejections_count,
                "biometric_reuse_detected": biometric_reuse_detected,
                "historical_risk_modifier": round(historical_risk_modifier, 2)
            },
            "trust_metrics": {
                "trust_score": round(trust_score, 2),
                "trust_factors": trust_factors
            },
            "threshold_policy": {
                "base_threshold": 45.0,
                "dynamic_threshold": dynamic_rejection_threshold,
                "reasoning": "; ".join(threshold_reasons)
            },
            "final_decision_reasoning": [
                f"Composite score is {composite_score:.2f} relative to dynamic threshold {dynamic_rejection_threshold:.1f}.",
                explanation
            ]
        }

        # Prepend the explanation and metrics payload to signals_summary
        final_summary = [
            {
                "category": "SYSTEM_SUMMARY",
                "severity": risk_level.value,
                "confidence": 1.0,
                "description": explanation,
                "fraud_payload": fraud_payload
            }
        ]
        final_summary.extend(signals_summary_list)

        return RiskEvaluation(
            case_id=case_id,
            composite_risk_score=round(composite_score, 2),
            risk_level=risk_level,
            triggered_signals_count=len(signals),
            signals_summary=final_summary,
            recommendation=recommendation
        )

    async def _generate_ai_risk_summary(self, signals: List[ThreatSignal], score: float) -> str:
        """
        Sends the list of detected threat signals to NVIDIA NIM to compile a natural-language reasoning summary.
        """
        if not settings.NVIDIA_APIKEY:
            return self._get_fallback_explanation(signals, score)
            
        try:
            headers = {
                "Authorization": f"Bearer {settings.NVIDIA_APIKEY}",
                "Content-Type": "application/json"
            }
            
            signals_data = []
            for s in signals:
                signals_data.append({
                    "engine": s.engine_name,
                    "category": s.category.value,
                    "severity": s.severity,
                    "confidence": s.confidence_score,
                    "description": s.description
                })

            prompt = (
                "You are an expert fraud risk auditor. Compile a concise, professional natural-language "
                "forensic reasoning summary of the case based on the following detected threat signals and composite score.\n\n"
                f"Composite Risk Score: {score}/100\n"
                f"Threat Signals List:\n{json.dumps(signals_data, indent=2)}\n\n"
                "Explain how the threats correlate (e.g. if document editing EXIF tags correlate with VLM layout tampering, "
                "or if synthetic voice coordinates with synthetic faces). Summarize the findings in 2-3 clear, authoritative sentences "
                "recommending the final action. Do not output any markdown formatting or prefix like 'Summary:'—just the direct explanation text."
            )

            payload = {
                "model": settings.MODELS["nvidia_nim"]["vlm_model"],
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.2
            }

            response = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.post(
                            "https://integrate.api.nvidia.com/v1/chat/completions",
                            headers=headers,
                            json=payload
                        )
                    if response.status_code == 200:
                        break
                    logger.warning(f"NVIDIA NIM risk scorer failed ({response.status_code}) on attempt {attempt + 1}: {response.text}")
                except httpx.HTTPError as exc:
                    logger.warning(f"HTTP connection/timeout error on attempt {attempt + 1}: {str(exc)}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
            
            if not response or response.status_code != 200:
                return self._get_fallback_explanation(signals, score)

            response_data = response.json()
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content")
            if not content:
                logger.error(f"NVIDIA NIM risk scorer returned empty or null content choice. Response payload: {response_data}")
                return self._get_fallback_explanation(signals, score)
            explanation = content.strip()
            return explanation

        except Exception as e:
            logger.error(f"NVIDIA NIM risk scorer failed: {str(e)}")
            return self._get_fallback_explanation(signals, score)

    def _get_fallback_explanation(self, signals: List[ThreatSignal], score: float) -> str:
        categories = list(set([s.category.value for s in signals]))
        return f"Case flagged with composite score of {score}. Major threat vectors identified: {', '.join(categories)}."
