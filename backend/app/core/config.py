# backend/app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    llm_provider: str = "deepseek"
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")

    embedding_provider: str = "local"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
    embedding_dimension: int = 768

    chroma_persist_directory: str = os.getenv(
        "CHROMA_PERSIST_DIRECTORY",
        "backend/data/chroma_db"
    )

    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    postgres_db: str = os.getenv("POSTGRES_DB", "deep_rag")

    deepseek_api_key: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url: str = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")


settings = Settings()