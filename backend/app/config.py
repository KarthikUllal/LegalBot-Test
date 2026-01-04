# backend/app/config.py
from pydantic_settings import BaseSettings
import os
from typing import Optional

class Settings(BaseSettings):
    DATA_DIR: str = "./datas"
    CHROMA_DIR: str = "./chroma_db"
    EMBED_MODEL: str= "models/embedding-001"

    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    BATCH_SIZE: int = 32
    ALLOW_ORIGINS: list = ["*"]
    OLLAMA_MODEL: str ="llama3.2:3b"
    FORCE_LOCAL: bool = False # make sure to change it to True when you dont use openRouter api key

    # NVIDIA Configuration
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_MODEL: str = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
    USE_NVIDIA: bool = os.getenv("USE_NVIDIA", "true").lower() == "true"

# OpenRouter Configuration (NOT USED - we only use NVIDIA NIM)
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: str = "google/gemini-2.0-flash-exp:free"
    USE_OPENROUTER: bool = False  # We are only using NVIDIA


    # TAVILY (web search fallback) - if not using, set to False
    TAVILY_API_KEY: Optional[str] = None
    USE_WEB_FALLBACK: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
