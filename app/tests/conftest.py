import pytest

from app.core.logger import logger
from app.core.database import db
from app.core.config import settings

TEST_COLLECTION_NAME = "test_rag_collection"


@pytest.fixture(autouse=True, scope="session")
def use_test_collection():
    original_collection_name = settings.COLLECTION_NAME
    settings.COLLECTION_NAME = TEST_COLLECTION_NAME

    yield

    settings.COLLECTION_NAME = original_collection_name


@pytest.fixture(autouse=True, scope="function")
async def create_test_collection():
    await db.initialize()

    yield

    try:
        await db.delete_collection()
    except Exception as e:
        logger.info(f"Test collection was not remove: {e}")
