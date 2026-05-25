from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from backend.app.core.config import settings
from sqlalchemy import text
from backend.app.models.db_models import Base

# Create async database engine
# Note: For SQLite fallback during rapid unit tests, we check the driver prefix
engine_url = settings.DATABASE_URL
if engine_url.startswith("postgresql://"):
    engine_url = engine_url.replace("postgresql://", "postgresql+asyncpg://")

# If using sqlite async, require sqllite+aiosqlite
if "sqlite" in engine_url and not engine_url.startswith("sqlite+aiosqlite://"):
    engine_url = engine_url.replace("sqlite://", "sqlite+aiosqlite://")

engine = create_async_engine(
    engine_url,
    echo=False,
    future=True,
    pool_pre_ping=True
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# Dependency injection generator for endpoints
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Initialization utility (useful for rapid startup / test environments)
async def init_db() -> None:
    async with engine.begin() as conn:
        # Create all tables if they don't exist
        await conn.run_sync(Base.metadata.create_all)
        
        # Add ocr_payload column if it's missing (robust schema migration)
        try:
            await conn.execute(text("ALTER TABLE cases ADD COLUMN ocr_payload JSON DEFAULT '{}';"))
        except Exception:
            pass

        # Add debug_images column if it's missing
        try:
            await conn.execute(text("ALTER TABLE cases ADD COLUMN debug_images JSON DEFAULT '[]';"))
        except Exception:
            pass
