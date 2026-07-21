"""
AuthBrain AI Face Analysis Engine
Configuration Management

Uses pydantic-settings for type-safe environment variable loading.
All settings can be overridden via .env file or environment variables.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=["../.env", ".env"],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    APP_NAME: str = "AuthBrain AI Face Analysis Engine"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "production", "testing"] = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    JWT_SECRET_KEY: str = "7a83d73507c570bbfbdc73f3cb6cfb0b30bb0443e067d022b7a37213ee8964d4"

    # ── Server ─────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1  # Keep at 1 — MediaPipe is not process-safe across workers

    # ── CORS ───────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"]
    )

    # ── Database ───────────────────────────────────────────────────────────────
    POSTGRES_USER: str = "authbrain"
    POSTGRES_PASSWORD: str = "authbrain_secret"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "authbrain_db"
    DATABASE_URL_OVERRIDE: str | None = Field(default=None, alias="DATABASE_URL")

    @computed_field  # type: ignore[misc]
    @property
    def DATABASE_URL(self) -> str:
        if self.DATABASE_URL_OVERRIDE:
            url = self.DATABASE_URL_OVERRIDE
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            elif url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            # Remove pgbouncer=true query parameter which asyncpg dialect doesn't accept
            url = url.replace("?pgbouncer=true", "").replace("&pgbouncer=true", "")
            return url
        if self.ENVIRONMENT == "development" and self.POSTGRES_HOST == "localhost":
            return "sqlite+aiosqlite:///authbrain_dev.db"
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field  # type: ignore[misc]
    @property
    def DATABASE_URL_SYNC(self) -> str:
        """Sync URL used by Alembic migrations."""
        if self.DATABASE_URL_OVERRIDE:
            url = self.DATABASE_URL_OVERRIDE
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+psycopg2://", 1)
            elif url.startswith("postgresql://") and not url.startswith("postgresql+psycopg2://"):
                url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
            return url
        if self.ENVIRONMENT == "development" and self.POSTGRES_HOST == "localhost":
            return "sqlite:///authbrain_dev.db"
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── MediaPipe / Analysis ───────────────────────────────────────────────────
    MAX_NUM_FACES: int = Field(default=4, ge=1, le=8)
    MIN_FACE_DETECTION_CONFIDENCE: float = Field(default=0.7, ge=0.0, le=1.0)
    MIN_FACE_PRESENCE_CONFIDENCE: float = Field(default=0.7, ge=0.0, le=1.0)
    MIN_TRACKING_CONFIDENCE: float = Field(default=0.5, ge=0.0, le=1.0)

    # Path to MediaPipe face landmarker .task model file
    FACE_LANDMARKER_MODEL_PATH: str = "models/face_landmarker.task"

    # ── Eye Analysis Thresholds ────────────────────────────────────────────────
    EAR_BLINK_THRESHOLD: float = Field(default=0.25, ge=0.1, le=0.5)
    EAR_BLINK_CONSEC_FRAMES: int = Field(default=3, ge=1, le=10)
    EAR_CLOSURE_THRESHOLD: float = Field(default=0.2, ge=0.1, le=0.5)

    # ── Mouth Analysis Thresholds ─────────────────────────────────────────────
    MAR_YAWN_THRESHOLD: float = Field(default=0.6, ge=0.3, le=0.9)
    MAR_YAWN_CONSEC_FRAMES: int = Field(default=15, ge=5, le=30)

    # ── Performance ────────────────────────────────────────────────────────────
    TARGET_FPS: int = Field(default=30, ge=10, le=60)
    FRAME_QUEUE_SIZE: int = Field(default=5, ge=1, le=20)
    JPEG_QUALITY: int = Field(default=80, ge=50, le=100)

    # ── Deep Learning Research Platform ───────────────────────────────────────
    # Master switch — set to True in .env to enable DL modules.
    # When False (default), the existing pipeline is completely unchanged.
    DL_ENABLED: bool = False

    # PyTorch compute device: "cpu" or "cuda" (auto-detected if available)
    DL_DEVICE: str = "cpu"

    # Comma-separated list of enabled emotion model IDs.
    # Available: "hsemotion", "deepface", "gnn_gat"
    DL_EMOTION_MODELS: list[str] = Field(default=["hsemotion"])
    DL_EMOTION_MIN_CONFIDENCE: float = Field(default=0.15, ge=0.0, le=1.0)

    # Enable Graph Neural Network inference
    DL_GNN_ENABLED: bool = True

    # Path to a trained GNN checkpoint. If missing, GNN inference is skipped.
    DL_GNN_CHECKPOINT_PATH: str = "models/dl/gnn_gat.pt"

    # Number of past frames fed to temporal models (LSTM / ST-GCN)
    DL_TEMPORAL_WINDOW: int = Field(default=30, ge=5, le=120)

    # Ensemble combination strategy: "weighted_avg" | "majority_vote" | "max_confidence"
    DL_ENSEMBLE_STRATEGY: str = "weighted_avg"

    # Enable expensive XAI explanations (GNNExplainer, SHAP)
    DL_XAI_ENABLED: bool = False

    # Local directory for caching downloaded model weights
    DL_MODEL_CACHE_DIR: str = "models/dl"

    # Per-model inference timeout: skip & log if exceeded
    DL_INFERENCE_TIMEOUT_MS: float = Field(default=500.0, ge=5.0, le=500.0)

    # Enable backend debug mode to archive frames, crops, and logits
    DL_DEBUG_MODE: bool = True

    # GNN graph edge construction strategy: "anatomical" | "knn" | "radius"
    DL_GRAPH_EDGE_STRATEGY: str = "knn"
    DL_GRAPH_KNN_K: int = Field(default=6, ge=2, le=20)

    # GAT hyperparameters
    DL_GAT_HIDDEN_CHANNELS: int = Field(default=64, ge=16, le=512)
    DL_GAT_HEADS: int = Field(default=4, ge=1, le=16)
    DL_GAT_LAYERS: int = Field(default=3, ge=1, le=8)

    # ── WebSocket ─────────────────────────────────────────────────────────────
    WS_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB max frame size
    WS_PING_INTERVAL: float = 20.0       # seconds
    WS_PING_TIMEOUT: float = 10.0        # seconds

    # ── Session ────────────────────────────────────────────────────────────────
    SESSION_TTL_SECONDS: int = 3600       # 1 hour
    PERSIST_ALL_FRAMES: bool = False      # Store every frame (high storage)
    PERSIST_SESSION_SUMMARY: bool = True  # Store session summaries

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return upper


@lru_cache
def get_settings() -> Settings:
    """Returns cached settings instance (singleton pattern)."""
    return Settings()
