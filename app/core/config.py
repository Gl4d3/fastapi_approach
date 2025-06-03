from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Kenyan Document Processor"
    VERSION: str = "2.0.0"
    DESCRIPTION: str = "AI-powered document validation and data extraction"
    
    # Server
    PORT: int = int(os.getenv("PORT", 8000))
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # File Processing
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_BATCH_SIZE: int = 50
    SUPPORTED_IMAGE_TYPES: List[str] = [
        "image/jpeg", "image/jpg", "image/png", "image/tiff", 
        "image/bmp", "image/webp", "image/gif"
    ]
    SUPPORTED_PDF_TYPES: List[str] = ["application/pdf"]
    
    # Redis - Fixed to include REDIS_URL
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    
    # OCR
    TESSERACT_PATH: str = "/usr/bin/tesseract"
    OCR_LANGUAGES: str = "eng"
    
    # Processing
    MAX_CONCURRENT_JOBS: int = 10
    JOB_TIMEOUT: int = 300  # 5 minutes
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    
    # Templates
    TEMPLATES_DIR: str = "templates"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings