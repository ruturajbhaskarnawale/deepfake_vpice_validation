from datetime import datetime, date
from enum import Enum
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

class ThreatCategory(str, Enum):
    DOCUMENT_TAMPERING = "DOCUMENT_TAMPERING"
    FACE_SPOOF = "FACE_SPOOF"
    DEEPFAKE_IMAGE = "DEEPFAKE_IMAGE"
    SYNTHETIC_VOICE = "SYNTHETIC_VOICE"
    IDENTITY_INCONSISTENCY = "IDENTITY_INCONSISTENCY"
    METADATA_ANOMALY = "METADATA_ANOMALY"
    VELOCITY_EXCEEDED = "VELOCITY_EXCEEDED"
    RAPID_RETRY_PATTERN = "RAPID_RETRY_PATTERN"

class ThreatSignal(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    engine_name: str
    category: ThreatCategory
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Engine's confidence in threat presence")
    severity: str = Field(..., description="LOW, MEDIUM, HIGH, CRITICAL")
    description: str
    evidence_payload: Dict[str, Any] = Field(default_factory=dict, description="Engine-specific metadata (e.g., coordinates, spectrogram hashes)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class RiskEvaluation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    case_id: UUID
    composite_risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_level: RiskLevel
    triggered_signals_count: int
    signals_summary: List[Dict[str, Any]] = Field(default_factory=list)
    recommendation: str = Field(..., description="APPROVE, REJECT, ESCALATE_TO_HUMAN")
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)

class IdentityNode(BaseModel):
    identity_id: UUID
    full_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    document_ids: List[str] = Field(default_factory=list)
    face_embeddings_hashes: List[str] = Field(default_factory=list)
    voice_embeddings_hashes: List[str] = Field(default_factory=list)
    associated_ips: List[str] = Field(default_factory=list)
    associated_emails: List[str] = Field(default_factory=list)
    risk_score: float = 0.0
    is_synthetic_suspect: bool = False

class AuditAction(str, Enum):
    INGESTED = "INGESTED"
    VALIDATED = "VALIDATED"
    ANALYSIS_COMPLETED = "ANALYSIS_COMPLETED"
    HUMAN_OVERRIDE = "HUMAN_OVERRIDE"
    SYSTEM_DECISION = "SYSTEM_DECISION"

class AuditTrail(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    case_id: UUID
    action: AuditAction
    actor: str = Field(..., description="System module name or Analyst User ID")
    details: str
    ip_address: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class EvidencePackage(BaseModel):
    case_id: UUID
    created_at: datetime
    sanitized_files: List[str] = Field(default_factory=list)
    metadata_forensics: Dict[str, Any] = Field(default_factory=dict)
    detected_threats: List[ThreatSignal] = Field(default_factory=list)
    risk_evaluation: Optional[RiskEvaluation] = None
    audit_history: List[AuditTrail] = Field(default_factory=list)

# Job Status Schemas
class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class IngestionResponse(BaseModel):
    case_id: UUID
    status: JobStatus
    message: str
    timestamp: datetime

class JobResponse(BaseModel):
    case_id: UUID
    status: JobStatus
    files_received: List[str]
    sanitized_files: List[str] = Field(default_factory=list)
    debug_images: List[str] = Field(default_factory=list)
    evidence: Optional[EvidencePackage] = None
    created_at: datetime
    updated_at: datetime
