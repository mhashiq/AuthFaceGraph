"""
AuthBrain AI Face Analysis Engine
Pytest Configuration & Fixtures

Configures in-memory SQLite database for tests, overrides dependencies,
and provides authenticated clients and seed helpers.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from fastapi.testclient import TestClient

import app.core.database
import app.core.config

# 1. Override init_db and close_db with no-ops for lifespan setup
async def dummy_init_db():
    pass
async def dummy_close_db():
    pass

app.core.database.init_db = dummy_init_db
app.core.database.close_db = dummy_close_db

# 2. Point to the downloaded model folder relative to backend directory
settings = app.core.config.get_settings()
settings.FACE_LANDMARKER_MODEL_PATH = "../models/face_landmarker.task"

from app.core.database import Base, get_db_session
from app.core.security import UserRole, create_access_token, hash_password
from app.main import app
from app.models.db_models import Organization, User

# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

TestingSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create a session-scoped event loop."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def init_test_db() -> AsyncGenerator[None, None]:
    """Initialize DB schema before testing."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()


@pytest.fixture(autouse=True)
async def clean_db() -> None:
    """Clear all database tables before running a test."""
    async with test_engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(table.delete())


@pytest.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    """Get clean DB session for each test."""
    async with TestingSessionLocal() as session:
        yield session


# Override DB dependency in FastAPI app
async def override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with TestingSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

app.dependency_overrides[get_db_session] = override_get_db_session


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Sync TestClient for WebSocket testing."""
    with TestClient(app) as tc:
        yield tc


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """AsyncClient for endpoint testing."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


@pytest.fixture
async def seed_data(db: AsyncSession) -> dict:
    """Seed test organization, users and return auth credentials & tokens."""
    # Organization
    org = Organization(
        name="Test Org",
        slug="test-org",
        is_active=True,
        plan="standard",
    )
    db.add(org)
    await db.flush()

    # Password hash
    pwd_hash = hash_password("secret_pass")

    # Admin User
    admin = User(
        org_id=org.id,
        email="admin@test.com",
        full_name="Admin User",
        password_hash=pwd_hash,
        role=UserRole.ADMINISTRATOR.value,
        is_active=True,
    )
    # Employee User
    employee = User(
        org_id=org.id,
        email="emp@test.com",
        full_name="Employee User",
        password_hash=pwd_hash,
        role=UserRole.EMPLOYEE.value,
        is_active=True,
    )
    db.add(admin)
    db.add(employee)
    await db.commit()

    # Tokens
    admin_token = create_access_token(
        user_id=str(admin.id),
        email=admin.email,
        role=UserRole.ADMINISTRATOR,
        org_id=str(org.id),
    )
    employee_token = create_access_token(
        user_id=str(employee.id),
        email=employee.email,
        role=UserRole.EMPLOYEE,
        org_id=str(org.id),
    )

    return {
        "org_id": str(org.id),
        "admin_user_id": str(admin.id),
        "admin_token": admin_token,
        "employee_user_id": str(employee.id),
        "employee_token": employee_token,
    }
