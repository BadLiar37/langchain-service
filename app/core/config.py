from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    OLLAMA_TIMEOUT: int = 30

    COLLECTION_NAME: str = "documents"
    CHROMA_HOST: str
    CHROMA_PORT: int = 8000

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 Mb

    APP_PORT: int = 8080
    APP_ENV: str = "development"

    LLM_CACHE_TTL: int
    LLM_CACHE_MAXSIZE: int

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
