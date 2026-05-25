from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from uuid import UUID
from backend.app.core.database import get_db
from backend.app.core.security import verify_api_key
from backend.app.models.db_models import Case
from backend.app.models.pydantic_models import (
    JobResponse, EvidencePackage, ThreatSignal, 
    RiskEvaluation, AuditTrail, JobStatus
)

router = APIRouter()

@router.get("/status/{case_id}", response_model=JobResponse)
async def get_case_status_and_evidence(
    case_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Retrieves the processing state of a case. Upon completion, packages 
    comprehensive forensic details, audit logs, and risk analysis metrics.
    """
    # Fetch Case eagerly loading children
    result = await db.execute(
        select(Case)
        .where(Case.id == case_id)
        .options(
            selectinload(Case.threat_signals),
            selectinload(Case.risk_evaluation),
            selectinload(Case.audit_logs)
        )
    )
    case = result.scalar_one_or_none()
    
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case ID {case_id} could not be resolved."
        )

    evidence_pkg = None

    # Package evidence if execution is completed
    if case.status == JobStatus.COMPLETED:
        # Convert DB signals to Pydantic
        signals = [
            ThreatSignal(
                id=s.id,
                engine_name=s.engine_name,
                category=s.category,
                confidence_score=s.confidence_score,
                severity=s.severity,
                description=s.description,
                evidence_payload=s.evidence_payload,
                timestamp=s.timestamp
            ) for s in case.threat_signals
        ]

        # Convert DB evaluation to Pydantic
        risk_eval = None
        if case.risk_evaluation:
            re = case.risk_evaluation
            risk_eval = RiskEvaluation(
                id=re.id,
                case_id=re.case_id,
                composite_risk_score=re.composite_risk_score,
                risk_level=re.risk_level,
                triggered_signals_count=re.triggered_signals_count,
                signals_summary=re.signals_summary,
                recommendation=re.recommendation,
                evaluated_at=re.evaluated_at
            )

        # Convert DB audit logs to Pydantic (sorted chronologically)
        sorted_db_audits = sorted(case.audit_logs, key=lambda x: x.timestamp)
        audits = [
            AuditTrail(
                id=au.id,
                case_id=au.case_id,
                action=au.action,
                actor=au.actor,
                details=au.details,
                ip_address=au.ip_address,
                timestamp=au.timestamp
            ) for au in sorted_db_audits
        ]

        evidence_pkg = EvidencePackage(
            case_id=case.id,
            created_at=case.created_at,
            sanitized_files=case.sanitized_files,
            metadata_forensics={
                "ingested_files_count": len(case.files_received),
                "sanitization_status": "SUCCESS",
                "ocr_payload": case.ocr_payload
            },
            detected_threats=signals,
            risk_evaluation=risk_eval,
            audit_history=audits
        )

    return JobResponse(
        case_id=case.id,
        status=case.status,
        files_received=[f for f in case.files_received],
        sanitized_files=[sf for sf in case.sanitized_files],
        debug_images=getattr(case, 'debug_images', []),
        evidence=evidence_pkg,
        created_at=case.created_at,
        updated_at=case.updated_at
    )
