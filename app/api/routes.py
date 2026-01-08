from fastapi import APIRouter, UploadFile, HTTPException
from app.core.logger import logger
from app.core.database import db
from app.models.schemas import (
    UploadResponse,
    AddChunksResponse,
    AddChunksRequest,
    AskRequest,
    AskResponse,
)
from app.services.chunking import chunking_service
from app.services.document_loader import DocumentLoader
from app.services.graph import langgraph_service
from app.services.pipeline import query_pipeline

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile):
    try:
        documents = await DocumentLoader.load_from_uploaded_file(file)
        chunks = await chunking_service.split_documents(documents)
        chunks_created = len(chunks)
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        await db.add_documents(texts, metadatas)
        logger.success(f"Processed and added {chunks_created} chunks to database")
        return UploadResponse(
            filename=file.filename,
            status="success",
            chunks_created=chunks_created,
            message="File uploaded successfully",
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/add-chunks", response_model=AddChunksResponse)
async def add_chunks(request: AddChunksRequest):
    try:
        chunks = await chunking_service.split_text(
            text=request.text, metadata=request.metadata or {}
        )
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        ids = await db.add_documents(texts, metadatas)

        return AddChunksResponse(status="success", chunks_added=len(ids), chunk_ids=ids)

    except Exception as e:
        logger.error(f"Add chunks failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask-question", response_model=AskResponse)
async def ask_question(request: AskRequest):
    try:
        result = await query_pipeline.ask(
            question=request.question,
            top_k=request.top_k,
            temperature=request.temperature,
            score_threshold=request.score_threshold,
        )

        return AskResponse(**result)

    except Exception as e:
        logger.error(f"Question processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask-graph", response_model=AskResponse)
async def ask_with_graph(request: AskRequest):
    try:
        result = await langgraph_service.process(
            query=request.question, top_k=request.top_k, temperature=request.temperature
        )

        result["metrics"] = {
            "query_type": result.pop("query_type", "unknown"),
            "documents_found": len(result.get("sources", [])),
        }
        return AskResponse(**result)

    except Exception as e:
        logger.error(f"Graph processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/db/stats")
async def db_stats():
    stats = await db.get_collection_stats()
    return stats
