# LangChain Document Service

Cистема для обработки документов с использованием LLM (Ollama), LangChain и ChromaDb.

## Быстрый старт

### 1. Запуск

```bash
# Запуск всех сервисов
docker-compose up --build -d

# Загрузка моделей Ollama
docker exec langchain_ollama ollama pull llama3
docker exec langchain_ollama ollama pull nomic-embed-text

# Проверка
curl http://localhost:8000/health
```

### 2. Загрузка документа

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@document.pdf"
```

### 3. Вопрос-ответ

```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is this document about?",
    "top_k": 4
  }'
```

## API Endpoints

### Работа с документами
- `POST /api/v1/upload` - загрузка файла
- `POST /api/v1/add-chunks` - добавление текста

### RAG (вопрос-ответ)
- `POST /api/v1/ask-question` - задать вопрос
  - `question` - вопрос (required)
  - `top_k` - количество документов (default: 4)
  - `temperature` - креативность (default: 0.7)
  - `score_threshold` - порог релевантности (default: 0.0)

### RAG (вопрос-ответ) с LangChain Graph
- `POST /api/v1/ask-graph` - задать вопрос с использованием графа: 
  1. Hello - будет обработан сценарий приветствия
  2. Search - будут обрадотан сценарий поиска
  3. Остальные - будет сценарий ответа на вопрос
  - `question` - вопрос (required)
  - `top_k` - количество документов (default: 4)
  - `temperature` - креативность (default: 0.7)
  - `score_threshold` - порог релевантности (default: 0.0)

### Служебные
- `GET /health` - проверка здоровья
- `GET /api/v1/stats` - статистика БД

**Swagger UI:** http://localhost:8000/docs

## Пример ответа

```json
{
  "answer": "Python is a high-level programming language...",
  "question": "What is Python?",
  "sources": [
    {
      "filename": "python_doc.pdf",
      "relevance_score": 0.89
    }
  ],
  "context_used": true,
  "model": "llama3",
  "metrics": {
    "search_time": 0.12,
    "generation_time": 2.34,
    "total_time": 2.46,
    "documents_found": 4
  }
}
```

## Структура проекта

```
app/
├── main.py                # FastAPI приложение
├── api/
│   └── routes.py          # API endpoints
├── core
│   ├── cache.py           # кэш ответов LLM модели
│   ├── config.py          # Конфигурация
│   ├── database.py        # ChromaDB
│   ├── logger.py          # Настройка Loguru
├── models/
│   └── schemas.py         # Pydantic модели
└── services/
    ├── chunking.py        # Разбивка на чанки
    ├── document_loader.py # Загрузка файлов
    ├── graph.py           # Langchain Graph
    ├── retrieval.py       # Поиск (similarity)
    ├── llm.py             # Ollama (промпты)
    └── pipeline.py        # RAG pipeline
```


## Примеры использования

### Базовый вопрос

```bash
curl -X POST "http://localhost:8000/api/v1/ask-question" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Python?",
    "top_k": 4
  }'
```

### С порогом релевантности

```bash
# Вернет только документы со score >= 0.5
curl -X POST "http://localhost:8000/api/v1/ask-question" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Python features",
    "score_threshold": 0.5
  }'
```

### Креативный ответ

```bash
# Высокая температура = более креативно
curl -X POST "http://localhost:8000/api/v1/ask-question" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Give examples of Python use",
    "temperature": 1.2
  }'
```
## Troubleshooting

### Ollama не отвечает

```bash
# Проверка
docker logs langchain_ollama
docker exec langchain_ollama ollama list

# Загрузка моделей
docker exec langchain_ollama ollama pull llama3
docker exec langchain_ollama ollama pull nomic-embed-text
```

### Нет релевантных документов

```bash
# Проверьте статистику
curl http://localhost:8000/api/v1/stats

```

### Ошибка embeddings

Проверьте, что модель для embeddings загружена:
```bash
docker exec langchain_ollama ollama list | grep nomic-embed-text
```
