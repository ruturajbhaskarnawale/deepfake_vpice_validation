import os
import uuid
import datetime
from typing import List
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.core.security import verify_api_key
from backend.app.core.orchestrator import CentralOrchestrator
from backend.app.models.db_models import Case
from backend.app.models.pydantic_models import IngestionResponse, JobStatus

router = APIRouter()
orchestrator = CentralOrchestrator()

# Define localized storage directories
STORAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage", "uploads"))

async def async_run_pipeline(case_id: uuid.UUID):
    """
    Asynchronous executor bridging background tasks with DB sessions.
    """
    from backend.app.core.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        try:
            await orchestrator.execute_pipeline(db, case_id)
        except Exception as e:
            # Main orchestration logs handled internally, background wrapper simply exits
            pass

@router.post("/upload", response_model=IngestionResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_media_payload(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Accepts files, persists them in localized folders, registers a Case index, 
    and triggers the asynchronous agent processing graphs.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one media asset must be supplied."
        )

    case_id = uuid.uuid4()
    case_upload_dir = os.path.join(STORAGE_DIR, str(case_id))
    os.makedirs(case_upload_dir, exist_ok=True)
    
    saved_file_paths = []
    
    # Save files to disk
    for file in files:
        # Standardize filenames
        clean_name = os.path.basename(file.filename)
        dest_path = os.path.join(case_upload_dir, clean_name)
        
        try:
            with open(dest_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_file_paths.append(dest_path)
        except Exception as e:
            # Clean up what was saved so far
            if os.path.exists(case_upload_dir):
                import shutil
                shutil.rmtree(case_upload_dir)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to persist asset '{clean_name}': {str(e)}"
            )

    # 1. Register Case relational state
    db_case = Case(
        id=case_id,
        status=JobStatus.PENDING,
        files_received=saved_file_paths,
        sanitized_files=[]
    )
    db.add(db_case)
    await db.commit()
    await db.refresh(db_case)

    # 2. Hand off to background execution pipeline
    background_tasks.add_task(async_run_pipeline, case_id)

    return IngestionResponse(
        case_id=case_id,
        status=JobStatus.PENDING,
        message="Payload successfully ingested and registered. Central Orchestration running asynchronously.",
        timestamp=datetime.datetime.utcnow()
    )
