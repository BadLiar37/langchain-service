import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from app.core.config import settings
from app.core.logger import logger


class VectorDatabase:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embeddings = None
        self.vectorstore = None

    async def initialize(self):
        try:
            self.client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            self.client.heartbeat()
            logger.info("Successfully connected to ChromaDB")
            self.embeddings = OllamaEmbeddings(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_EMBEDDING_MODEL,
            )
            logger.info(f"Embedding model:{settings.OLLAMA_EMBEDDING_MODEL}")

            collection_name = settings.COLLECTION_NAME
            existing_collections = [col.name for col in self.client.list_collections()]

            if collection_name in existing_collections:
                logger.info(f"Collection '{collection_name}' already exists")
            else:
                logger.info(f"Creating new collection '{collection_name}'")
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings,
            )

            logger.info("VectorStore initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def add_documents(self, documents: list, metadatas: list = None):
        try:
            if not self.vectorstore:
                raise ValueError("VectorStore not initialized")
            ids = await self.vectorstore.aadd_texts(
                texts=documents, metadatas=metadatas
            )
            logger.info(f"Added {len(ids)} documents to database")
            return ids

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    async def similarity_search(self, query: str, k: int = 4):
        try:
            if not self.vectorstore:
                raise ValueError("VectorStore not initialized")

            results = await self.vectorstore.asimilarity_search(query=query, k=k)

            return results

        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            raise

    async def get_collection_stats(self):
        try:
            collection = self.client.get_collection(settings.COLLECTION_NAME)
            count = collection.count()

            return {
                "collection_name": settings.COLLECTION_NAME,
                "document_count": count,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    async def delete_collection(self):
        try:
            self.client.delete_collection(settings.COLLECTION_NAME)
            logger.warning(f"Collection '{settings.COLLECTION_NAME}' deleted")
            await self.initialize()
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise


db = VectorDatabase()
