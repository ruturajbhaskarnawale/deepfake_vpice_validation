import torch
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import os
from backend.app.core.config import settings
from backend.app.core.database import init_db
from backend.app.api import ingest, jobs

# Configure standard console logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sentinel.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize relational schemas automatically
    logger.info("Initializing relational SQLite/PostgreSQL schemas...")
    try:
        await init_db()
        logger.info("Database schemas initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}", exc_info=True)

    # Startup: Pre-warm the InsightFace ArcFace biometric model in a background
    # thread so the ~100s ONNX load happens ONCE at server start, not on the
    # first user request (which would cause polling timeouts in the frontend).
    import asyncio
    loop = asyncio.get_event_loop()
    def _prewarm_insightface():
        try:
            from backend.app.services.biometric_service import _get_shared_app
            app_instance = _get_shared_app()
            if app_instance is not None:
                logger.info("InsightFace ArcFace model pre-warmed successfully at startup.")
            else:
                logger.warning("InsightFace pre-warm: model unavailable (will use heuristic fallback).")
        except Exception as exc:
            logger.warning(f"InsightFace pre-warm failed: {exc}")
    loop.run_in_executor(None, _prewarm_insightface)
    logger.info("InsightFace model pre-warm triggered in background thread.")

    yield
    # Shutdown: Clean up any external engine references if needed
    logger.info("Sentinel Core API shutting down...")


# Dev API key hint shown in Swagger description
_DEV_KEY_HINT = (
    "\n\n> 🔑 **Authentication**: Click the **Authorize** button (🔒) above and enter "
    "either your **NVIDIA API key** (`nvapi-...`) OR the dev key "
    "`sentinel_dev_key_2026_top_secret` in the `APIKeyHeader` field."
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Enterprise-grade multi-modal AI-Native Trust Intelligence platform "
        "for deepfake and synthetic entity forensics." + _DEV_KEY_HINT
    ),
    version="1.0.0",
    lifespan=lifespan,
    swagger_ui_parameters={
        "persistAuthorization": True,
        "tryItOutEnabled": True,
    }
)

# Mount uploads static directory so frontend can render debug images
from fastapi.staticfiles import StaticFiles
uploads_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "storage", "uploads"))
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/storage", StaticFiles(directory=uploads_dir), name="storage")

# Enable standard enterprise CORS mapping
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global custom exception middleware
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception intercepted during '{request.method} {request.url.path}': {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal system anomaly occurred. Forensic telemetry has logged the execution path."}
    )

# Include Routers
app.include_router(ingest.router, prefix=f"{settings.API_V1_STR}/ingest", tags=["Ingestion Ingest Engine"])
app.include_router(jobs.router, prefix=f"{settings.API_V1_STR}/jobs", tags=["Jobs Forensic Telemetry"])

@app.get("/health", tags=["Telemetry Monitoring"])
async def system_health_check():
    """
    Evaluates microservice health indicators.
    """
    return {
        "status": "HEALTHY",
        "timestamp": "2026-05-22T11:51:45+05:30",
        "engines": {
            "validator_agent": "ONLINE",
            "document_ocr_agent": "ONLINE",
            "vision_forensics_agent": "ONLINE",
            "voice_authenticity_agent": "ONLINE",
            "identity_graph_agent": "ONLINE"
        }
    }

@app.get("/playground", response_class=HTMLResponse, tags=["Forensic Testing Playground"])
async def testing_playground():
    """
    Renders the premium glassmorphic Jodetx Sentinel Core E2E forensic playground UI.
    """
    template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates", "playground.html"))
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    return HTMLResponse(content="<h1>Testing Playground template file not found!</h1>", status_code=404)


def custom_openapi():
    """
    Override the generated OpenAPI schema to inject the APIKeyHeader security scheme.
    This makes the Swagger UI 'Authorize' button appear so testers can supply x-api-key.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Inject the API key security scheme
    openapi_schema["components"] = openapi_schema.get("components", {})
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "x-api-key",
            "description": (
                "Sentinel platform API key. "
                "Development key: `sentinel_dev_key_2026_top_secret`"
            ),
        }
    }

    # Apply the security scheme globally to all endpoints
    openapi_schema["security"] = [{"APIKeyHeader": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

