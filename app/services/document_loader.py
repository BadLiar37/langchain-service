import hashlib
import os
import tempfile
from pathlib import Path

from fastapi import UploadFile
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredEPubLoader,
)
from langchain_community.document_loaders.parsers.audio import FasterWhisperParser
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain_core.documents import Document
from app.core.logger import logger


class DocumentLoader:
    LOADER_MAPPING = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".md": UnstructuredMarkdownLoader,
        ".epub": UnstructuredEPubLoader,
        ".mp3": "audio",
        ".wav": "audio",
        ".m4a": "audio",
        ".ogg": "audio",
        ".flac": "audio",
    }

    @classmethod
    def _get_file_hash(cls, file_path: Path) -> str:
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Hash was not calculated {file_path}: {e}")
            return ""

    @classmethod
    def _enrich_metadata(
        cls, docs: list[Document], file_path: Path, original_name: str | None = None
    ) -> None:
        file_name = original_name or file_path.name
        file_stats = file_path.stat()

        file_hash = cls._get_file_hash(file_path)

        for i, doc in enumerate(docs):
            doc.metadata.update(
                {
                    "source": file_name,
                    "file_type": file_path.suffix.lower()[1:],
                    "file_path": str(file_path),
                    "file_size": file_stats.st_size,
                    "created_at": file_stats.st_ctime,
                    "modified_at": file_stats.st_mtime,
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "total_chunks": len(docs),
                }
            )
            # for .pdf, .docx
            if "page" in doc.metadata:
                doc.metadata["page_number"] = doc.metadata["page"] + 1

    @classmethod
    async def load_document(cls, file_path: str | Path) -> list[Document] | None:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()

        if extension not in cls.LOADER_MAPPING:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {list(cls.LOADER_MAPPING.keys())}"
            )

        try:
            if extension in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
                parser = FasterWhisperParser(
                    model_size="tiny",
                    device="cpu",
                )
                blob_loader = FileSystemBlobLoader(
                    str(file_path.parent), glob=file_path.name
                )
                loader = GenericLoader(blob_loader, parser)

                documents = loader.load()
                cls._enrich_metadata(documents, file_path)
                for doc in documents:
                    doc.metadata.update(
                        {
                            "transcription": True,
                            "transcription_model": "faster-whisper",
                            "transcription_model_size": "tiny",
                        }
                    )

                logger.info(
                    f"Audio transcribed: {file_path.name} → {len(documents)} segments"
                )
            else:
                loader_class = cls.LOADER_MAPPING[extension]
                loader = loader_class(str(file_path))

                documents = loader.load()
                cls._enrich_metadata(documents, file_path)
                logger.info(
                    f"Successfully loaded {len(documents)} pages from {file_path.name}"
                )
            return documents
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            raise

    @classmethod
    async def load_from_uploaded_file(
        cls,
        uploaded_file: UploadFile,
        original_filename: str | None = None,
    ) -> list[Document]:
        filename = original_filename or uploaded_file.filename or "unknown"
        suffix = Path(filename).suffix.lower()

        if suffix not in cls.LOADER_MAPPING:
            supported = ", ".join(cls.LOADER_MAPPING.keys())
            raise ValueError(
                f"Unsupported format: {suffix}. Supported formats are: {supported}"
            )

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            content = await uploaded_file.read()
            tmp_file.write(content)
            tmp_file.close()

            file_path = Path(tmp_file.name)
            documents = await cls.load_document(file_path)
            for doc in documents:
                doc.metadata["source"] = filename
            logger.info(
                f"The document was uploaded successfully: {filename} → {len(documents)} parts"
            )
            return documents

        except Exception as e:
            logger.error(f"Error while uploading a document {filename}: {e}")
            raise
        finally:
            try:
                if Path(tmp_file.name).exists():
                    os.unlink(tmp_file.name)
            except Exception as cleanup_error:
                logger.warning(
                    f"Temporary file was not removed {tmp_file.name}: {cleanup_error}"
                )

    @classmethod
    async def get_supported_formats(cls) -> list[str]:
        return list(cls.LOADER_MAPPING.keys())
