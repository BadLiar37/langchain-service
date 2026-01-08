from typing import Any
from app.core.logger import logger

from langchain_core.documents import Document

from app.core.database import db


class RetrievalService:
    def __init__(self):
        self.vectorstore = None

    async def search(
        self, query: str, k: int = 4, score_threshold: float = 0.0
    ) -> list[Document]:
        try:
            if not self.vectorstore:
                self.vectorstore = db.vectorstore

            if not self.vectorstore:
                raise ValueError("VectorStore not initialized")

            logger.info(f"Searching for: '{query}' (top_k={k})")

            results = await self.vectorstore.asimilarity_search_with_relevance_scores(
                query=query, k=k, score_threshold=score_threshold
            )

            documents = []
            for doc, score in results:
                doc.metadata["relevance_score"] = score
                documents.append(doc)
                logger.debug(
                    f"Found: {doc.metadata.get('filename', 'unknown')} "
                    f"(score={score:.3f})"
                )

            logger.info(f"Found {len(documents)} relevant documents")
            return documents

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def format_context(self, documents: list[Document]) -> str:
        if not documents:
            return "No relevant information found."

        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("filename", "Unknown source")
            score = doc.metadata.get("relevance_score")

            header = f"[Source {i}: {source}"
            if score:
                header += f" - relevance: {score:.2f}"
            header += "]"

            context_parts.append(f"{header}\n{doc.page_content}\n")

        return "\n---\n".join(context_parts)

    def get_sources(self, documents: list[Document]) -> list[dict[str, Any]]:
        sources = []

        for doc in documents:
            sources.append(
                {
                    "filename": doc.metadata.get("source", "Unknown"),
                    "file_type": doc.metadata.get("file_type", ""),
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                    "relevance_score": doc.metadata.get("relevance_score"),
                }
            )

        return sources


retrieval_service = RetrievalService()
