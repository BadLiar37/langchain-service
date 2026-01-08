from pathlib import Path
from app.services.document_loader import DocumentLoader


async def test_load_txt(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello, this is test document!\nSecond row")

    docs = await DocumentLoader.load_document(str(file_path))

    assert len(docs) == 1
    assert "Hello" in docs[0].page_content
    assert docs[0].metadata["source"] == "test.txt"
    assert docs[0].metadata["file_type"] == "txt"


async def test_get_supported_formats():
    formats = await DocumentLoader.get_supported_formats()
    expected = [
        ".txt",
        ".pdf",
        ".docx",
        ".md",
        ".epub",
        ".mp3",
        ".wav",
        ".m4a",
        ".ogg",
        ".flac",
    ]
    assert set(formats) == set(expected)
