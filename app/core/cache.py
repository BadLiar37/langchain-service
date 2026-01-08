from cachetools import TTLCache
from hashlib import sha256
import json

from app.core.config import settings


llm_response_cache = TTLCache(
    maxsize=settings.LLM_CACHE_MAXSIZE, ttl=settings.LLM_CACHE_TTL
)


def get_llm_cache_key(question: str, context: str, temperature: float) -> str:
    data = {
        "question": question.strip(),
        "context": context.strip(),
        "temperature": round(temperature, 3),
    }
    json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return sha256(json_str.encode("utf-8")).hexdigest()
