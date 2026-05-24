import logging
import json
import asyncio
from typing import Any, Dict, List
import httpx
from backend.app.agents.base import BaseAgent
from backend.app.core.config import settings
from backend.app.models.pydantic_models import RiskEvaluation, RiskLevel, ThreatSignal

logger = logging.getLogger("sentinel.risk_scorer")

class RiskScorerAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "RiskScorerAgent"

    async def process(self, case_id: str, context: Dict[str, Any]) -> List[ThreatSignal]:
        # The risk scorer outputs a RiskEvaluation object, which is appended to the context.
        # It does not create individual threat signals.
        return []

    async def evaluate_risk(self, case_id: Any, signals: List[ThreatSignal]) -> RiskEvaluation:
        """
        Compiles individual threat signals, applies custom weight multipliers, 
        calls NVIDIA NIM to write a natural language explanation, and calculates 
        a consolidated RiskEvaluation.
        """
        if not signals:
            return RiskEvaluation(
                case_id=case_id,
                composite_risk_score=5.0, # Baseline benign check score
                risk_level=RiskLevel.LOW,
                triggered_signals_count=0,
                signals_summary=[{
                    "category": "SYSTEM_SUMMARY",
                    "severity": "LOW",
                    "confidence": 1.0,
                    "description": "All multi-modal verification verification checks completed successfully. No threat vectors detected."
                }],
                recommendation="APPROVE"
            )

        # Weighted calculation table
        total_weight = 0.0
        signals_summary = []

        for signal in signals:
            sev = signal.severity.upper()
            weight = 5.0
            if sev == "CRITICAL":
                weight = 45.0
            elif sev == "HIGH":
                weight = 30.0
            elif sev == "MEDIUM":
                weight = 15.0
                
            # Apply confidence scaling factor
            weighted_contribution = weight * signal.confidence_score
            total_weight += weighted_contribution
            
            signals_summary.append({
                "category": signal.category.value,
                "severity": signal.severity,
                "confidence": signal.confidence_score,
                "description": signal.description
            })

        # Cap overall score at 100.0
        composite_score = min(total_weight, 100.0)

        # Enforce threshold-based score override for CRITICAL signals
        critical_signals = [s for s in signals if s.severity.upper() == "CRITICAL"]
        if critical_signals:
            max_critical_conf = max(s.confidence_score for s in critical_signals)
            if max_critical_conf >= 0.7:
                override_score = 75.0
            else:
                override_score = 50.0
            composite_score = max(composite_score, override_score)

        # Determine consolidated classification and recommendation thresholds
        if composite_score >= 70.0:
            risk_level = RiskLevel.CRITICAL
            recommendation = "REJECT"
        elif composite_score >= 45.0:
            risk_level = RiskLevel.HIGH
            recommendation = "REJECT"
        elif composite_score >= 20.0:
            risk_level = RiskLevel.MEDIUM
            recommendation = "ESCALATE_TO_HUMAN"
        else:
            risk_level = RiskLevel.LOW
            recommendation = "APPROVE"

        # Generate the natural language explanation via NIM LLM
        explanation = await self._generate_ai_risk_summary(signals, composite_score)

        # Prepend the explanation to signals_summary
        final_summary = [{
            "category": "SYSTEM_SUMMARY",
            "severity": risk_level.value,
            "confidence": 1.0,
            "description": explanation
        }]
        final_summary.extend(signals_summary)

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
        Sends the list of detected threat signals to NVIDIA NIM to compile a natural language reasoning summary.
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
                "model": "meta/llama-3.2-11b-vision-instruct",
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
            explanation = response_data["choices"][0]["message"]["content"].strip()
            return explanation

        except Exception as e:
            logger.error(f"NVIDIA NIM risk scorer failed: {str(e)}")
            return self._get_fallback_explanation(signals, score)

    def _get_fallback_explanation(self, signals: List[ThreatSignal], score: float) -> str:
        categories = list(set([s.category.value for s in signals]))
        return f"Case flagged with composite score of {score}. Major threat vectors identified: {', '.join(categories)}."
