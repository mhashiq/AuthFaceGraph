"""
AuthBrain AI Face Analysis Engine
Async Database Engine & Session Management

Uses SQLAlchemy 2.0 async ORM with asyncpg driver.
Sessions are managed via dependency injection in FastAPI routes.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


# Create async engine with dialect-aware connection options
_engine_kwargs = {
    "echo": settings.DEBUG,
    "pool_pre_ping": True,
}
if settings.DATABASE_URL.startswith("postgresql"):
    _engine_kwargs.update({
        "pool_size": 10,
        "max_overflow": 20,
        "pool_recycle": 3600,
        "connect_args": {
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
        },
    })

engine = create_async_engine(
    settings.DATABASE_URL,
    **_engine_kwargs,
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,    # Keep data accessible after commit
    autocommit=False,
    autoflush=False,
)


async def init_db() -> None:
    """
    Initialize database tables on startup.
    In production, use Alembic migrations instead.
    """
    async with engine.begin() as conn:
        # Import all models to register them with Base.metadata
        from app.models import db_models  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_initialized", url=settings.POSTGRES_HOST)

    # Seed default organization and admin user if empty
    from sqlalchemy import select
    from app.models.db_models import Organization, User
    from app.core.security import hash_password
    async with AsyncSessionLocal() as session:
        try:
            org_stmt = select(Organization).limit(1)
            result = await session.execute(org_stmt)
            if not result.scalar():
                logger.info("seeding_default_organization_and_user")
                org = Organization(
                    name="Default Org",
                    slug="default",
                    is_active=True
                )
                session.add(org)
                await session.flush()

                admin = User(
                    org_id=org.id,
                    email="admin@authbrain.com",
                    full_name="System Administrator",
                    password_hash=hash_password("password123"),
                    role="administrator",
                    is_active=True
                )
                session.add(admin)
                await session.commit()
                logger.info("seeding_complete", email=admin.email)
        except Exception as exc:
            logger.warning("database_seeding_failed", error=str(exc))


async def close_db() -> None:
    """Dispose engine connection pool on shutdown."""
    await engine.dispose()
    logger.info("database_connection_closed")


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides an async database session.
    Automatically handles commit on success and rollback on error.

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions in non-FastAPI contexts
    (background tasks, tests, etc.).
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
