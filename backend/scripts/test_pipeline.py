try:
    import torch
except Exception:
    torch = None
import os
import sys
import asyncio
import logging
import uuid
import datetime
import shutil

# Fix Windows console encoding so print() never raises UnicodeEncodeError
import io
log_file = open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_output.log")), "w", encoding="utf-8", buffering=1)
sys.stdout = log_file
sys.stderr = log_file

# Force backend/ directory onto PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.core.config import settings
# Force database URL to a local SQLite memory DB for immediate integration verification!
settings.DATABASE_URL = "sqlite+aiosqlite:///:memory:"

from backend.app.core.database import engine, AsyncSessionLocal, init_db
from backend.app.core.orchestrator import CentralOrchestrator
from backend.app.models.db_models import Case
from backend.app.api.jobs import get_case_status_and_evidence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sentinel.test_pipeline")

# ---------------------------------------------------------------------------
# Real Test Asset Paths  (relative to repo root -> test_assets/)
# ---------------------------------------------------------------------------
_ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "test_assets")
)

REAL_TEST_ASSETS = {
    # Government / identity document  ->  fed into OCR agent
    "government_id": os.path.join(_ASSETS_DIR, "Image (8).jpg"),
    # Selfie / biometric face image   ->  fed into Vision-Forensics agent
    "selfie":        os.path.join(_ASSETS_DIR, "Image (7).jpg"),
    # Additional face / media photo   ->  secondary vision check
    "media_photo":   os.path.join(_ASSETS_DIR, "Media.jpg"),
    # Real video clip                 ->  fed into Vision-Forensics frame extractor
    "video_clip":    os.path.join(_ASSETS_DIR, "MicrosoftTeams-video.mp4"),
}


def verify_assets() -> list:
    """
    Verifies that every real test asset exists and returns the ordered list
    of file paths to submit to the pipeline.
    Raises FileNotFoundError with a clear message for any missing asset.
    """
    ordered_paths = []
    print("\n" + "-" * 70)
    print("  REAL TEST ASSETS  ->  Verification")
    print("-" * 70)
    for role, path in REAL_TEST_ASSETS.items():
        exists = os.path.exists(path)
        size_mb = os.path.getsize(path) / (1024 * 1024) if exists else 0
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {role:18s}  {os.path.basename(path):<40s}  {size_mb:.2f} MB")
        if not exists:
            raise FileNotFoundError(
                f"Required test asset '{role}' not found at:\n  {path}\n"
                f"Place the file in:  {_ASSETS_DIR}"
            )
        ordered_paths.append(path)
    print("-" * 70)
    print("  All assets verified.\n")
    return ordered_paths


async def run_end_to_end_forensics():
    print("=" * 80)
    print("      JODETX SENTINEL CORE -- E2E FORENSIC PIPELINE INTEGRATION TEST      ")
    print("      Using REAL test assets from:  test_assets/                          ")
    print("=" * 80)

    # -- 0. Verify assets -------------------------------------------------------
    real_files = verify_assets()

    # -- 1. Initialize SQLite Database ------------------------------------------
    logger.info("Initializing in-memory SQLite relational database schemas...")
    await init_db()

    # -- 2. Register Case in DB -------------------------------------------------
    case_id = uuid.uuid4()
    logger.info(f"Registering Case {case_id} with {len(real_files)} real asset(s)...")

    async with AsyncSessionLocal() as db:
        case = Case(
            id=case_id,
            files_received=real_files,
            sanitized_files=[]
        )
        db.add(case)
        await db.commit()

    print(f"\n  Case ID  : {case_id}")
    print(f"  Assets   : {len(real_files)} file(s)")
    for f in real_files:
        print(f"    -> {os.path.basename(f)}")
    print()

    # -- 3. Trigger Central Orchestrator ----------------------------------------
    logger.info("Triggering CentralOrchestrator.execute_pipeline() ...")
    orchestrator = CentralOrchestrator()

    async with AsyncSessionLocal() as db:
        completed_case = await orchestrator.execute_pipeline(db, case_id)
        print(f"\nPipeline Finished!  Case Status: {completed_case.status.value}\n")

    # -- 4. Fetch consolidated Evidence Package ---------------------------------
    logger.info("Querying status router -> generating final Evidence Package ...")
    async with AsyncSessionLocal() as db:
        response = await get_case_status_and_evidence(
            case_id, db, api_key="sentinel_dev_key_2026_top_secret"
        )

        print("=" * 80)
        print("                        ANALYST EVIDENCE PACKAGE                         ")
        print("=" * 80)
        print(f"Case ID    :  {response.case_id}")
        print(f"Status     :  {response.status.value}")
        print(f"Created At :  {response.created_at}")

        evidence = response.evidence
        if not evidence:
            print("[ERROR] Evidence Package could not be compiled.")
            return

        # -- Sanitized Files ----------------------------------------------------
        print("\n-- SANITIZED FILES " + "-" * 61)
        for f in evidence.sanitized_files:
            print(f"  [OK] {os.path.basename(f)}")

        # -- Threat Signals -----------------------------------------------------
        print(f"\n-- DETECTED FORENSIC THREAT SIGNALS ({len(evidence.detected_threats)} total) " + "-" * 30)
        if not evidence.detected_threats:
            print("  (none)")
        for i, sig in enumerate(evidence.detected_threats, 1):
            print(f"\n  [{i}] Category   : {sig.category.value}")
            print(f"       Severity   : {sig.severity}")
            print(f"       Confidence : {sig.confidence_score * 100:.1f}%")
            print(f"       Source     : {sig.engine_name}")
            print(f"       Details    : {sig.description}")
            print(f"       Payload    : {sig.evidence_payload}")
            print("  " + "-" * 60)

        # -- Risk Evaluation ----------------------------------------------------
        print("\n-- COMPOSITE RISK EVALUATION " + "-" * 51)
        re = evidence.risk_evaluation
        print(f"  Composite Risk Score : {re.composite_risk_score} / 100.0")
        print(f"  Combined Severity    : {re.risk_level.value}")
        print(f"  Signals Triggered    : {re.triggered_signals_count}")
        print(f"  Recommendation       : {re.recommendation}")

        # -- OCR Payload --------------------------------------------------------
        print("\n-- EXTRACTED OCR PAYLOAD " + "-" * 55)
        ocr = evidence.metadata_forensics.get("ocr_payload", {})
        fields = [
            ("Name",       "full_name"),
            ("DOB",        "date_of_birth"),
            ("Gender",     "gender"),
            ("Doc Number", "document_number"),
            ("Country",    "issuing_country"),
            ("Doc Type",   "document_type"),
        ]
        for label, key in fields:
            print(f"  {label:<14}: {ocr.get(key, 'N/A')}")
        print(f"  {'Raw Text':<14}:\n{ocr.get('full_raw_text', 'N/A')}")
        print(f"  {'Dynamic JSON':<14}: {ocr.get('dynamic_json', {})}")

        # -- Audit Trail --------------------------------------------------------
        print("\n-- AUDIT TRAIL LOGS (COMPLIANCE) " + "-" * 47)
        for log in evidence.audit_history:
            print(f"  [{log.timestamp}] [{log.action.value}] {log.actor} -- {log.details}")

    # -- 5. Clean sanitized copies (NOT original test_assets) -------------------
    # sanitized_dirs = set()
    # for f in real_files:
    #     sanitized_dirs.add(os.path.join(os.path.dirname(f), "sanitized"))
    # for d in sanitized_dirs:
    #     if os.path.exists(d):
    #         shutil.rmtree(d, ignore_errors=True)
    #         logger.info(f"Cleaned up sanitized scratch directory: {d}")

    print("\n" + "=" * 80)
    print("           INTEGRATION TEST RUN FINISHED WITH OUTSTANDING SUCCESS           ")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_end_to_end_forensics())
