from typing import Any
from app.core.logger import logger
import time

from app.services.retrieval import retrieval_service
from app.services.llm import llm_service


class QueryPipeline:
    def __init__(self):
        self.retrieval = retrieval_service
        self.llm = llm_service

    async def ask(
        self,
        question: str,
        top_k: int = 4,
        temperature: float = 0.7,
        score_threshold: float = 0.0,
    ) -> dict[str, Any]:
        start_time = time.time()

        try:
            logger.info(f"Processing question: '{question[:100]}...'")
            logger.info(f"Parameters: top_k={top_k}, temperature={temperature}")

            search_start = time.time()
            documents = await self.retrieval.search(
                query=question, k=top_k, score_threshold=score_threshold
            )
            search_time = time.time() - search_start

            logger.info(
                f"Search completed in {search_time:.2f}s, found {len(documents)} documents"
            )

            if not documents:
                return {
                    "answer": "I couldn't find any relevant information in the database to answer your question.",
                    "question": question,
                    "sources": [],
                    "context_used": False,
                    "metrics": {
                        "search_time": search_time,
                        "generation_time": 0,
                        "total_time": time.time() - start_time,
                        "documents_found": 0,
                    },
                }

            context = self.retrieval.format_context(documents)
            sources = self.retrieval.get_sources(documents)

            logger.info(
                f"Context formatted: {len(context)} chars from {len(sources)} sources"
            )

            generation_start = time.time()
            response = await self.llm.generate_answer(
                question=question, context=context, temperature=temperature
            )
            generation_time = time.time() - generation_start

            logger.info(f"Answer generated in {generation_time:.2f}s")

            total_time = time.time() - start_time

            result = {
                "answer": response["answer"],
                "question": question,
                "sources": sources,
                "context_used": True,
                "model": response["model"],
                "metrics": {
                    "search_time": search_time,
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "documents_found": len(documents),
                    "context_length": len(context),
                },
            }

            logger.info(f"Question processed successfully in {total_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)

            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "question": question,
                "sources": [],
                "context_used": False,
                "error": str(e),
                "metrics": {
                    "total_time": time.time() - start_time,
                    "documents_found": 0,
                },
            }


query_pipeline = QueryPipeline()
