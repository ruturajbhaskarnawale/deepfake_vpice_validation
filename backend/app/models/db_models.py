import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import String, Float, DateTime, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from backend.app.models.pydantic_models import JobStatus, RiskLevel, ThreatCategory, AuditAction

class Base(DeclarativeBase):
    pass

class Case(Base):
    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    status: Mapped[JobStatus] = mapped_column(SQLEnum(JobStatus), default=JobStatus.PENDING)
    files_received: Mapped[List[str]] = mapped_column(JSON, default=list)
    sanitized_files: Mapped[List[str]] = mapped_column(JSON, default=list)
    ocr_payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    threat_signals: Mapped[List["ThreatSignalDB"]] = relationship(
        "ThreatSignalDB", back_populates="case", cascade="all, delete-orphan"
    )
    risk_evaluation: Mapped[Optional["RiskEvaluationDB"]] = relationship(
        "RiskEvaluationDB", back_populates="case", uselist=False, cascade="all, delete-orphan"
    )
    audit_logs: Mapped[List["AuditLogDB"]] = relationship(
        "AuditLogDB", back_populates="case", cascade="all, delete-orphan"
    )

class ThreatSignalDB(Base):
    __tablename__ = "threat_signals"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"))
    engine_name: Mapped[str] = mapped_column(String(100))
    category: Mapped[ThreatCategory] = mapped_column(SQLEnum(ThreatCategory))
    confidence_score: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(20))  # LOW, MEDIUM, HIGH, CRITICAL
    description: Mapped[str] = mapped_column(String(500))
    evidence_payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    case: Mapped["Case"] = relationship("Case", back_populates="threat_signals")

class RiskEvaluationDB(Base):
    __tablename__ = "risk_evaluations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), unique=True)
    composite_risk_score: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[RiskLevel] = mapped_column(SQLEnum(RiskLevel))
    triggered_signals_count: Mapped[int] = mapped_column(default=0)
    signals_summary: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    recommendation: Mapped[str] = mapped_column(String(50))  # APPROVE, REJECT, ESCALATE_TO_HUMAN
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    case: Mapped["Case"] = relationship("Case", back_populates="risk_evaluation")

class AuditLogDB(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"))
    action: Mapped[AuditAction] = mapped_column(SQLEnum(AuditAction))
    actor: Mapped[str] = mapped_column(String(100))  # System component or user ID
    details: Mapped[str] = mapped_column(String(1000))
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    case: Mapped["Case"] = relationship("Case", back_populates="audit_logs")
