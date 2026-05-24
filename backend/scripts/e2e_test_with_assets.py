#!/usr/bin/env python3
"""End‑to‑end forensic pipeline test using supplied test assets.
The script:
1. Switches the DB to an in‑memory SQLite instance.
2. Creates a Case row containing the four test asset files.
3. Executes the CentralOrchestrator pipeline.
4. Queries the case status/evidence endpoint and prints the JSON payload.
"""

import os
import sys
import asyncio
import uuid
import logging
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.core.config import settings
# Force in‑memory DB for a clean run
settings.DATABASE_URL = "sqlite+aiosqlite:///:memory:"

from backend.app.core.database import init_db, AsyncSessionLocal
from backend.app.core.orchestrator import CentralOrchestrator
from backend.app.models.db_models import Case
from backend.app.api.jobs import get_case_status_and_evidence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sentinel.e2e_test")

async def main():
    logger.info("Initializing DB schema...")
    await init_db()

    # Locate the test asset directory
    assets_dir = Path("c:/Users/rutur/OneDrive/Desktop/deepfake/test_assets")
    asset_files = [str(p) for p in assets_dir.iterdir() if p.is_file()]
    logger.info(f"Using assets: {asset_files}")

    case_id = uuid.uuid4()
    async with AsyncSessionLocal() as db:
        case = Case(id=case_id, files_received=asset_files, sanitized_files=[])
        db.add(case)
        await db.commit()

    orchestrator = CentralOrchestrator()
    async with AsyncSessionLocal() as db:
        completed_case = await orchestrator.execute_pipeline(db, case_id)
        logger.info(f"Pipeline finished – status: {completed_case.status.value}")

    async with AsyncSessionLocal() as db:
        response = await get_case_status_and_evidence(case_id, db, api_key="sentinel_dev_key_2026_top_secret")
        # Print the full JSON response
        import json
        print(json.dumps(response.dict(), indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())
