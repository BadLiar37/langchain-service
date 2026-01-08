from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.logger import logger
from app.core.config import settings


class ChunkingService:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info(
            f"ChunkingService initialized: chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

    async def split_documents(self, documents: list[Document]) -> list[Document]:
        try:
            chunks = self.text_splitter.split_documents(documents)

            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i
                chunk.metadata["chunk_size"] = len(chunk.page_content)
                if "filename" not in chunk.metadata:
                    logger.warning(f"Chunk {i} missing 'filename' metadata!")
                    chunk.metadata["filename"] = "Unknown"

                if "file_type" not in chunk.metadata:
                    logger.warning(f"Chunk {i} missing 'file_type' metadata!")
                    chunk.metadata["file_type"] = ""

                logger.debug(
                    f"Chunk {i}: filename={chunk.metadata.get('filename')}, "
                    f"file_type={chunk.metadata.get('file_type')}, "
                    f"size={chunk.metadata.get('chunk_size')}"
                )

            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")

            return chunks

        except Exception as e:
            logger.error(f"Failed to split documents: {e}")
            raise

    def get_optimal_chunk_size(self, text_length: int) -> int:
        if text_length < 1000:
            return 256
        elif text_length < 5000:
            return 512
        elif text_length < 20000:
            return 1024
        else:
            return 2048


chunking_service = ChunkingService()
