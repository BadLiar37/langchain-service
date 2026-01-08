from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from starlette.requests import Request

from app.api.routes import router
from app.core.logger import logger
from app.core.database import db
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")
    logger.info(f"Environment: {settings.APP_ENV}")
    logger.info(f"Ollama URL: {settings.OLLAMA_BASE_URL}")
    logger.info(f"ChromaDB: {settings.CHROMA_HOST}:{settings.CHROMA_PORT}")

    try:
        await db.initialize()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    yield

    logger.info("Shutting down application...")


app = FastAPI(
    title="LangChain Document Service",
    description="Service for document processing with LLM and vector search",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response


app.include_router(router, prefix="/api/v1", tags=["documents"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "langchain-document-service",
        "environment": settings.APP_ENV,
    }
