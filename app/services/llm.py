from typing import Any

from app.core.cache import get_llm_cache_key, llm_response_cache
from app.core.logger import logger
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

from app.core.config import settings


class LLMService:
    def __init__(self):
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        try:
            logger.info(f"Initializing Ollama LLM: {settings.OLLAMA_MODEL}")
            logger.info(f"Ollama URL: {settings.OLLAMA_BASE_URL}")

            self.llm = Ollama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                temperature=0.7,
                timeout=settings.OLLAMA_TIMEOUT,
            )

            logger.info("✅ Ollama LLM initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            raise

    def get_prompt_template(self) -> PromptTemplate:
        template = """You are a helpful AI assistant. Use the following context to answer the user's question.
If you cannot find the answer in the context, say so honestly. Do not make up information.

Context:
{context}

Question: {question}

Answer:"""

        return PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

    async def generate_answer(
        self, question: str, context: str, temperature: float = 0.7
    ) -> dict[str, Any]:
        try:
            logger.info(f"Generating answer for: '{question[:50]}...'")

            cache_key = get_llm_cache_key(question, context, temperature)

            if cache_key in llm_response_cache:
                cached = llm_response_cache[cache_key]
                logger.success(f"LLM answer from cache! (key: {cache_key[:8]}...)")
                return cached

            prompt = self.get_prompt_template()

            chain = LLMChain(llm=self.llm, prompt=prompt)

            original_temp = self.llm.temperature
            self.llm.temperature = temperature

            result = await chain.ainvoke({"context": context, "question": question})

            self.llm.temperature = original_temp

            answer = result.get("text", "").strip()

            logger.info(f"Answer generated: {answer[:100]}...")

            response_data = {
                "answer": answer,
                "question": question,
                "model": settings.OLLAMA_MODEL,
                "temperature": temperature,
            }
            llm_response_cache[cache_key] = response_data

            return response_data

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise

    async def test_connection(self) -> bool:
        try:
            response = await self.llm.ainvoke("Hello, respond with 'OK'")
            logger.info(f"Test response: {response}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


llm_service = LLMService()
