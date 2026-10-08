import os
from pathlib import Path
from typing import Dict, Any
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

def load_yaml(file_path: Path) -> Dict[str, Any]:
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

class Settings(BaseSettings):
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    log_level: str = Field(default="info", validation_alias="LOG_LEVEL")

    # Database connections
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")

    # Inference Backends
    local_slm_url: str = Field(default="http://localhost:8001/v1", validation_alias="LOCAL_SLM_URL")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    deepseek_api_key: str = Field(default="", validation_alias="DEEPSEEK_API_KEY")
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    zep_api_key: str = Field(default="", validation_alias="ZEP_API_KEY")
    zep_api_url: str = Field(default="https://api.getzep.com", validation_alias="ZEP_API_URL")

    # Gateway Parameters & Thresholds
    exact_cache_enabled: bool = Field(default=True, validation_alias="EXACT_CACHE_ENABLED")
    exact_cache_ttl_seconds: int = Field(default=86400, validation_alias="EXACT_CACHE_TTL_SECONDS")
    semantic_cache_enabled: bool = Field(default=True, validation_alias="SEMANTIC_CACHE_ENABLED")
    semantic_cache_threshold: float = Field(default=0.85, validation_alias="SEMANTIC_CACHE_THRESHOLD")
    compression_enabled: bool = Field(default=True, validation_alias="COMPRESSION_ENABLED")
    compression_ratio: float = Field(default=0.80, validation_alias="COMPRESSION_RATIO")
    monthly_budget_limit_usd: float = Field(default=500.0, validation_alias="MONTHLY_BUDGET_LIMIT_USD")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
config_yaml = load_yaml(BASE_DIR / "config" / "config.yaml")
models_yaml = load_yaml(BASE_DIR / "config" / "models.yaml")
