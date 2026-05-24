# 🛡️ Jodetx Sentinel Core

> **Enterprise-Grade Multi-Modal AI Trust Intelligence Platform for Deepfake & Synthetic Identity Forensics**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://python.org)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM%20API-76B900?style=flat&logo=nvidia)](https://integrate.api.nvidia.com)
[![SQLite](https://img.shields.io/badge/Database-SQLite%20%7C%20PostgreSQL-003B57?style=flat&logo=sqlite)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-Proprietary-red?style=flat)](.)

---

## 📋 Table of Contents

- [What Is This?](#-what-is-this)
- [Core Capabilities](#-core-capabilities)
- [System Architecture](#-system-architecture)
- [AI Models & NVIDIA NIM](#-ai-models--nvidia-nim)
- [Agent Pipeline](#-agent-pipeline)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Environment Configuration](#-environment-configuration)
- [API Reference](#-api-reference)
- [Playground UI](#-playground-ui)
- [Data Models](#-data-models)
- [Security](#-security)
- [Database](#-database)
- [Testing](#-testing)

---

## 🎯 What Is This?

**Jodetx Sentinel Core** is an AI-native forensic intelligence backend that automatically detects:

- 🖼️ **Deepfake images** — GAN/Diffusion-generated synthetic faces in selfies and documents
- 📄 **Document tampering** — Forged IDs, edited PDFs, font/layout inconsistencies
- 🎤 **Synthetic voice** — TTS-cloned audio, vocoder artifacts, missing glottal pulses
- 🧑 **Identity inconsistencies** — Age/gender mismatches between declared fields and biometrics
- 🔗 **Cross-case identity fraud** — Face/voice embeddings reused under different names

The system is built for **KYC (Know Your Customer)** pipelines, financial onboarding platforms, and trust & safety teams who need to automatically screen submitted media assets for synthetic or tampered content before approving identity claims.

---

## ✨ Core Capabilities

| Capability | Description |
|---|---|
| **Multi-Modal Analysis** | Processes images, documents (PDF), and audio in a single pipeline |
| **NVIDIA NIM Integration** | Uses `meta/llama-3.2-11b-vision-instruct` VLM via NVIDIA's API Gateway |
| **Async Agent Architecture** | All detection stages run as independent async agents with shared context |
| **Structured Evidence Packaging** | Every case produces a full evidence package with threat signals, risk scores, and audit trail |
| **Composite Risk Scoring** | Weighted signal aggregation produces a 0–100 composite score with APPROVE/REJECT/ESCALATE decision |
| **Forensic Playground UI** | A live glassmorphic web UI for drag-and-drop testing without writing any code |
| **Swagger API** | Full OpenAPI 3.0 documentation with pre-configured authentication |
| **Audit Logging** | Every pipeline stage is logged with actor, action, and timestamp |
| **SQLite (Dev) / PostgreSQL (Prod)** | Works out-of-the-box locally with SQLite; scales to PostgreSQL via Docker |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                                    │
│   Playground UI (/playground)  │  Swagger UI (/docs)  │  REST API   │
└────────────────────┬────────────────────────────────────────────────┘
                     │ HTTP POST /api/v1/ingest/upload
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI GATEWAY (main.py)                         │
│  • CORS Middleware  • API Key Authentication  • Global Exc. Handler  │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│              CENTRAL ORCHESTRATOR (orchestrator.py)                  │
│                                                                      │
│  STAGE 1          STAGE 2                STAGE 3       STAGE 4      │
│  Validation  →  [OCR Agent]          →  Identity  →  Risk Scorer    │
│  & Sanitize  →  [Vision Agent]          Graph                       │
│               →  [Voice Agent]                                       │
│                                                                      │
│              All agents share a mutable context dict                 │
└────────────────────┬────────────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌─────────────────┐   ┌─────────────────────────────┐
│  NVIDIA NIM API │   │     SQLite / PostgreSQL DB   │
│  integrate.api. │   │  Cases, ThreatSignals,       │
│  nvidia.com     │   │  RiskEvaluation, AuditLogs   │
└─────────────────┘   └─────────────────────────────┘
```

---

## 🤖 AI Models & NVIDIA NIM

All intelligence extraction is powered by **NVIDIA NIM (NVIDIA Inference Microservices)** via the API gateway at `https://integrate.api.nvidia.com`.

### Model Used

| Model | Provider | Purpose |
|---|---|---|
| `meta/llama-3.2-11b-vision-instruct` | Meta / NVIDIA | Document OCR + Layout Forensics + Deepfake & Liveness Detection |

### How It Works

The VLM (Vision Language Model) is invoked with **structured forensic prompts** that instruct it to return a strict JSON block. Two distinct prompt strategies are used:

**Document OCR Prompt** — Extracts:
- `extracted_fields`: `full_name`, `date_of_birth`, `gender`, `document_number`, `issuing_country`, `document_type`
- `tamper_score`: Float 0.0–1.0 representing layout tampering probability
- `evidence_summary`: Human-readable layout analysis
- `layout_anomalies`: List of specific detected anomalies

**Vision Forensics Prompt** — Extracts:
- `biometric_findings`: `estimated_age`, `estimated_gender`
- `deepfake_score`: Float 0.0–1.0 representing GAN/synthetic probability
- `liveness_score`: Float 0.0–1.0 (1.0 = natural, 0.0 = printed/screen spoof)
- `visual_anomalies`: List of detected image artifacts

### Retry Strategy
All NVIDIA NIM calls use **3-attempt exponential backoff** (2s, 4s delays) with `httpx.AsyncClient` and a 60-second timeout per attempt.

### Fallback Parser
If the VLM responds with markdown/conversational text instead of a JSON block, a **regex-based fallback parser** extracts all structured fields from the natural language response.

---

## 🔄 Agent Pipeline

The pipeline is orchestrated by `CentralOrchestrator` and runs **5 sequential stages**:

### Stage 1 — ValidatorAgent
**Purpose**: File validation and media sanitization before any analysis.

- ✅ File size check (max 50 MB)
- ✅ Extension allowlist: `jpg`, `jpeg`, `png`, `pdf`, `mp4`, `wav`, `mp3`
- ✅ MIME type verification (via `python-magic` or `mimetypes` fallback)
- ✅ Image corruption check via PIL `verify()`
- ✅ Minimum resolution enforcement (400×400 px)
- ✅ PDF header signature validation (`%PDF`)
- ✅ WAV RIFF container signature check
- ✅ Media normalization (RGB conversion, copy to `sanitized/` directory)

### Stage 2a — DocumentOCRAgent
**Purpose**: Extract identity fields from ID documents and detect layout tampering.

- Calls `meta/llama-3.2-11b-vision-instruct` via NVIDIA NIM with a structured OCR + forensics prompt
- Parses `extracted_fields` (name, DOB, gender, doc number, country, type)
- Generates `DOCUMENT_TAMPERING` threat signal if `tamper_score > 0.4`
- Locally checks EXIF metadata for editing software signatures (Photoshop, GIMP, etc.)
- Populates `ocr_payload` in shared context for downstream agents

### Stage 2b — VisionForensicsAgent
**Purpose**: Detect deepfakes, GAN artifacts, and face liveness spoofing.

- Calls `meta/llama-3.2-11b-vision-instruct` via NVIDIA NIM with a visual forensics prompt
- Generates `DEEPFAKE_IMAGE` signal if `deepfake_score > 0.4`
- Generates `FACE_SPOOF` signal if `liveness_score < 0.6`
- Extracts biometric estimates (`estimated_age`, `estimated_gender`) into shared context
- Generates a SHA-256-based face embedding fingerprint for identity graph lookups

### Stage 2c — VoiceAuthenticityAgent
**Purpose**: Detect TTS-cloned or synthetic audio.

- Processes `.wav` and `.mp3` files
- Detects synthetic voice markers (vocoder artifact patterns, missing glottal pulses)
- Generates `SYNTHETIC_VOICE` threat signals with spectrogram consistency scores
- Generates voice embedding fingerprints for identity graph lookups
- *(Production: integrates NVIDIA Riva/NeMo for real spectrogram analysis)*

### Stage 3 — IdentityGraphAgent
**Purpose**: Cross-modal identity correlation and demographic plausibility checks.

- Compares declared DOB age vs. VLM biometric estimated age
- Flags `IDENTITY_INCONSISTENCY` if deviation exceeds **15 years**
- Performs graph-based duplicate detection on face/voice embedding fingerprints
- *(Production: integrates Neo4j Cypher query: `MATCH (i:Identity)-[:HAS_FACE]->(f:FaceEmbedding {hash: $hash}) RETURN i`)*

### Stage 4 — RiskScorerAgent
**Purpose**: Aggregate all threat signals into a unified risk evaluation.

**Weighted scoring table:**

| Severity | Weight per Signal |
|---|---|
| CRITICAL | 45.0 |
| HIGH | 30.0 |
| MEDIUM | 15.0 |
| LOW | 5.0 |

Each signal's contribution = `weight × confidence_score`. All contributions are summed and capped at 100.0.

**Decision thresholds:**

| Score Range | Risk Level | Recommendation |
|---|---|---|
| ≥ 70.0 | CRITICAL | REJECT |
| 45.0 – 69.9 | HIGH | REJECT |
| 20.0 – 44.9 | MEDIUM | ESCALATE_TO_HUMAN |
| < 20.0 | LOW | APPROVE |

---

## 📁 Project Structure

```
deepfake/
├── .env                          # API keys & database URL
├── docker-compose.yml            # PostgreSQL + Redis services
├── sentinel_local.db             # SQLite local development database
└── backend/
    ├── requirements.txt          # Python dependencies
    ├── storage/
    │   └── uploads/              # Uploaded files per case (UUID folders)
    ├── scripts/
    │   └── test_pipeline.py      # End-to-end integration test script
    └── app/
        ├── main.py               # FastAPI app, Swagger config, routes
        ├── agents/
        │   ├── base.py           # Abstract BaseAgent class
        │   ├── validator.py      # Stage 1: File validation & sanitization
        │   ├── document_ocr.py   # Stage 2a: NVIDIA NIM OCR + layout forensics
        │   ├── vision_forensics.py # Stage 2b: NVIDIA NIM deepfake + liveness
        │   ├── voice_auth.py     # Stage 2c: Synthetic voice detection
        │   ├── identity_graph.py # Stage 3: Identity correlation & graph checks
        │   └── risk_scorer.py    # Stage 4: Composite risk scoring
        ├── api/
        │   ├── ingest.py         # POST /api/v1/ingest/upload
        │   └── jobs.py           # GET /api/v1/jobs/status/{case_id}
        ├── core/
        │   ├── config.py         # Pydantic Settings (reads .env)
        │   ├── database.py       # SQLAlchemy async engine & session
        │   ├── orchestrator.py   # CentralOrchestrator pipeline
        │   └── security.py       # API key verification
        ├── models/
        │   ├── db_models.py      # SQLAlchemy ORM models
        │   └── pydantic_models.py # Pydantic request/response schemas
        └── templates/
            └── playground.html   # Glassmorphic forensic playground UI
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- NVIDIA API Key from [build.nvidia.com](https://build.nvidia.com)

### 1. Clone and navigate
```bash
cd deepfake
```

### 2. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Configure environment
```bash
# Edit .env file:
NVIDIA_APIKEY=nvapi-your-key-here
DATABASE_URL=sqlite+aiosqlite:///sentinel_local.db
```

### 4. Start the server
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload



### 5. Open the Playground
Navigate to **http://localhost:8000/playground**

Or test via the **Swagger UI** at **http://127.0.0.1:8000/docs**

---

## ⚙️ Environment Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `NVIDIA_APIKEY` | **Yes** | — | NVIDIA NIM API key (`nvapi-...`) from build.nvidia.com |
| `DATABASE_URL` | No | PostgreSQL URI | `sqlite+aiosqlite:///sentinel_local.db` for local dev |

### Production (Docker)
```bash
docker-compose up -d       # Starts PostgreSQL + Redis
# Then set DATABASE_URL=postgresql+asyncpg://sentinel_user:sentinel_password@localhost:5432/sentinel_db
```

---

## 📡 API Reference

### Authentication
All endpoints require an `x-api-key` header. Two valid keys are accepted:
- Your **NVIDIA API key**: `nvapi-...` (same key from your `.env`)
- Dev key: `sentinel_dev_key_2026_top_secret`

### Endpoints

#### `POST /api/v1/ingest/upload`
Upload one or more media files for forensic analysis.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `files` | File[] | One or more files (JPG, PNG, PDF, WAV, MP3) |

**Headers:**
```
x-api-key: nvapi-your-key-here
Content-Type: multipart/form-data
```

**Response `202 Accepted`:**
```json
{
  "case_id": "57c0bf38-8a61-4811-8ab2-7c9f5e24ffd6",
  "status": "PENDING",
  "message": "Payload successfully ingested and registered. Central Orchestration running asynchronously.",
  "timestamp": "2026-05-22T08:30:00Z"
}
```

---

#### `GET /api/v1/jobs/status/{case_id}`
Poll the processing status of a case. Returns full evidence package when `COMPLETED`.

**Response `200 OK`:**
```json
{
  "case_id": "57c0bf38-...",
  "status": "COMPLETED",
  "files_received": ["/path/to/uploaded/file.jpg"],
  "sanitized_files": ["/path/to/sanitized/sanitized_file.jpg"],
  "evidence": {
    "case_id": "57c0bf38-...",
    "detected_threats": [
      {
        "engine_name": "VisionForensicsAgent",
        "category": "DEEPFAKE_IMAGE",
        "confidence_score": 0.87,
        "severity": "CRITICAL",
        "description": "NVIDIA NIM visual forensics flagged image as synthetic deepfake.",
        "evidence_payload": {
          "nvidia_model": "meta/llama-3.2-11b-vision-instruct",
          "vlm_deepfake_score": 0.87,
          "detected_anomalies": ["smoothed skin texture", "eye asymmetry"]
        }
      }
    ],
    "risk_evaluation": {
      "composite_risk_score": 87.5,
      "risk_level": "CRITICAL",
      "recommendation": "REJECT",
      "triggered_signals_count": 2
    },
    "audit_history": [...]
  }
}
```

#### `GET /health`
System health check — verifies all agent engines are online.

#### `GET /playground`
Renders the interactive HTML forensic testing UI.

#### `GET /docs`
Swagger UI with interactive API testing (authentication pre-configured).

---

## 🖥️ Playground UI

The playground at `/playground` provides a premium glassmorphic interface for testing the system without writing any code.

**Features:**
- 📥 Drag & drop file upload
- 🚀 One-click forensic pipeline execution
- 💻 Live orchestration log stream
- 🛡️ Composite risk gauge (0–100 score visualization)
- ⚠️ Threat signal cards with severity badges
- 📝 Extracted OCR fields display
- ⏳ Pipeline audit trail timeline
- Pre-filled API key (no manual entry needed)

---

## 📊 Data Models

### ThreatSignal
```python
{
  "id": UUID,
  "engine_name": str,           # Which agent detected this
  "category": ThreatCategory,   # DEEPFAKE_IMAGE | DOCUMENT_TAMPERING | FACE_SPOOF | SYNTHETIC_VOICE | IDENTITY_INCONSISTENCY | METADATA_ANOMALY
  "confidence_score": float,    # 0.0 – 1.0
  "severity": str,              # LOW | MEDIUM | HIGH | CRITICAL
  "description": str,
  "evidence_payload": dict,     # Agent-specific metadata
  "timestamp": datetime
}
```

### RiskEvaluation
```python
{
  "composite_risk_score": float,  # 0.0 – 100.0
  "risk_level": RiskLevel,        # LOW | MEDIUM | HIGH | CRITICAL
  "recommendation": str,          # APPROVE | REJECT | ESCALATE_TO_HUMAN
  "triggered_signals_count": int,
  "signals_summary": list[dict]
}
```

---

## 🔐 Security

| Mechanism | Implementation |
|---|---|
| **API Key Auth** | `x-api-key` header validated via HMAC constant-time comparison |
| **Valid Keys** | NVIDIA API key (`nvapi-...`) or `sentinel_dev_key_2026_top_secret` |
| **MIME Validation** | Magic bytes inspection on every uploaded file |
| **File Size Limits** | 50 MB maximum per file |
| **Extension Allowlist** | Only `jpg, jpeg, png, pdf, mp4, wav, mp3` accepted |
| **CORS** | Configured (all origins for dev; restrict in production) |
| **JWT** | Infrastructure in place (`python-jose`) for future auth extension |

---

## 🗄️ Database

### SQLite (Development — Default)
```
DATABASE_URL=sqlite+aiosqlite:///sentinel_local.db
```
- Zero configuration, auto-initialized on first startup
- Database file: `sentinel_local.db` at project root

### PostgreSQL (Production)
```
DATABASE_URL=postgresql+asyncpg://sentinel_user:sentinel_password@localhost:5432/sentinel_db
```
Start with Docker:
```bash
docker-compose up -d postgres
```

### Schema Tables

| Table | Description |
|---|---|
| `cases` | Root entity per forensic submission. Holds status, file paths, OCR payload |
| `threat_signals` | Individual forensic findings from each agent |
| `risk_evaluations` | Final aggregated risk score and recommendation per case |
| `audit_logs` | Chronological pipeline action log per case |

---

## 🧪 Testing

### End-to-End Integration Test
```bash
python backend/scripts/test_pipeline.py
```

This script:
1. Creates synthetic test images (ID document + selfie + audio files)
2. Posts them to the live API
3. Polls for completion
4. Prints the full evidence package

### Manual Swagger Testing
1. Open **http://127.0.0.1:8000/docs**
2. Click 🔒 **Authorize**
3. Enter your NVIDIA key (`nvapi-...`) or `sentinel_dev_key_2026_top_secret`
4. Use `POST /api/v1/ingest/upload` → upload any image
5. Copy the `case_id` from the response
6. Use `GET /api/v1/jobs/status/{case_id}` to poll results

---

## 🛣️ Production Roadmap

| Phase | Feature |
|---|---|
| **Planned** | Neo4j graph database for real identity cross-case matching |
| **Planned** | NVIDIA Riva/NeMo for spectrogram-based voice deepfake detection |
| **Planned** | Video frame-level deepfake detection (temporal consistency analysis) |
| **Planned** | Webhook callbacks on pipeline completion |
| **Planned** | Multi-tenant API key management |
| **Planned** | Redis-backed async job queue for horizontal scaling |

---

## 📄 License

Proprietary — Jodetx Intelligence Platform  
All rights reserved © 2026
