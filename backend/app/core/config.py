# backend/app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ================== LLM 配置（只用 DeepSeek）==================
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"

    # ================== Embedding 配置（改成本地免费）==================
    embedding_provider: str = "local"               # ← 关键修改
    embedding_model: str = "BAAI/bge-small-zh-v1.5"           # 中文 RAG 最强免费模型
    embedding_dimension: int = 512                 # bge-m3 固定维度

    # ================== 向量库 & DB 配置 ==================
    chroma_persist_directory: str = "backend/data/chroma_db"   # 使用相对于项目根目录的路径

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "deep_rag"

    # ================== API Keys ==================
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"

settings = Settings()