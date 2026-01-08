from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
prefix = "/api/v1"


def test_upload_txt(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Text file for uploading")

    with open(file_path, "rb") as f:
        response = client.post(
            prefix + "/upload", files={"file": ("sample.txt", f, "text/plain")}
        )

    assert response.status_code == 200
    data = response.json()
    assert "status" in data or "answer" in data
    stats = client.get(prefix + "/db/stats").json()
    assert stats["document_count"] > 0


async def test_ask_question():
    response = client.post(
        prefix + "/ask-question", json={"question": "Describe me this document"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 10


async def test_unsupported_format(tmp_path):
    file_path = tmp_path / "bad.exe"
    file_path.write_bytes(b"fake binary")

    with open(file_path, "rb") as f:
        response = client.post(
            prefix + "/upload",
            files={"file": ("bad.exe", f, "application/octet-stream")},
        )

    assert response.status_code == 400
    assert "Unsupported format" in response.text
