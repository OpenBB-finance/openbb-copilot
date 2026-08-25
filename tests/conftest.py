import io
import os
import tempfile
import uuid
from enum import Enum
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import httpx
import pytest
import pytest_asyncio
import xxhash
from fastapi.testclient import TestClient
from openbb_ai.models import SourceInfo

from openbb_ada.constants import (
    OPENBB_PAYMENTS_API_SECRET_KEY,
    OPENBB_PAYMENTS_BASE_URL,
)
from openbb_ada.copilot import CopilotService
from openbb_ada.dependencies import (
    get_sql_agent_service,
    get_user_file_service,
)
from openbb_ada.main import app
from openbb_ada.models import (
    Document,
    UnavailableDocument,
    UrlFileReference,
    UserFile,
)
from openbb_ada.services import (
    ClientFunctionCallService,
    DocumentAgentService,
    DocumentService,
    LoggingService,
    NativeFunctionCallService,
    SqlAgentService,
    SqlQueryGenerationService,
    TemplateService,
    UserFileService,
)
from openbb_ada.utils.utils import (
    set_copilot_call_count,
)
from openbb_ada.vector_db import VectorDb, VectorDbDocument


@pytest.fixture
def test_database_engine():
    """
    Provide a test database engine with proper SQLite configuration for pytest.

    This fixture ensures thread-safe SQLite usage during tests by:
    - Using named in-memory DB ("sqlite://") instead of anonymous (":memory:")
    - Disabling check_same_thread for multi-threaded test scenarios
    - Using StaticPool to maintain single connection across threads
    """
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite://",  # Named in-memory DB
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    yield engine
    engine.dispose()  # Clean up after tests


class MockUUIDs(Enum):
    ID1 = "08d47a2f-bd35-4f53-a0e6-a45b4c7252f0"
    ID2 = "a1b2c3d4-1fbc-472a-852a-cc96adff0701"
    ID3 = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
    ID4 = "5815d0c5-e83a-4098-8f66-707e874c891b"
    ID5 = "e47ac10b-58cc-4372-a567-0e02b2c3d479"
    ID6 = "ea09cd3a-cc6a-4b02-8032-7bf0b542a0c8"
    ID7 = "266f83d5-c6c2-462c-9eee-7f08cff197e4"
    ID8 = "a87ecde4-c296-40fc-aaf8-a2b6e2daffdb"
    ID9 = "7b9a773f-bed7-495f-97c1-e026e10eaf93"
    ID10 = "11242bf8-02c3-497f-830e-957c8bf4fdb3"
    ID11 = "f49b59ff-ec8b-4437-8367-f1d5948c2750"
    ID12 = "3f26673e-bb28-488a-bb0c-c3a75229eed4"
    # Add more IDs if needed


@pytest.fixture
def mock_uuids() -> type[MockUUIDs]:
    return MockUUIDs


@pytest.fixture(autouse=True)
def reset_sse_starlette_appstatus_event():
    """
    Fixture that resets the appstatus event in the sse_starlette app.

    Should be used on any test that uses sse_starlette to stream events.
    """
    # See https://github.com/sysid/sse-starlette/issues/59
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None


@pytest.fixture
def vector_db_target_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_headers(valid_access_token):
    user_access_token = str(uuid.uuid4())
    yield {
        "Authorization": f"Bearer {user_access_token}",
        "X-User-Id": "test-user-id",
    }


@pytest.fixture
def valid_access_token():
    with patch(
        "openbb_ada.main.validate_and_sync",
        new_callable=AsyncMock,
        return_value=(True, []),
    ):
        yield


@pytest.fixture
def invalid_access_token():
    with patch(
        "openbb_ada.main.validate_access_token",
        new_callable=AsyncMock,
        return_value=False,
    ):
        yield


@pytest.fixture
def actual_valid_access_token() -> str:
    # NB: Only to be used for integration tests
    response = httpx.post(
        OPENBB_PAYMENTS_BASE_URL + "/pro/login",
        headers={"X-OpenBB-Authorization": f"Bearer {OPENBB_PAYMENTS_API_SECRET_KEY}"},
        json={
            "email": os.environ["OPENBB_TEST_EMAIL"],
            "password": os.environ["OPENBB_TEST_PASSWORD"],
        },
    )
    try:
        access_token = response.json()["access_token"]
    except KeyError as err:
        raise ValueError(
            f"Failed to retrieve access token from response: {response.text}"
        ) from err
    return access_token


@pytest.fixture
def mock_headers_with_actual_valid_access_token(actual_valid_access_token: str) -> dict:
    return {
        "Authorization": f"Bearer {actual_valid_access_token}",
        "X-User-Id": "test-user-id",
    }


@pytest_asyncio.fixture
async def actual_set_copilot_call_count(actual_valid_access_token: str):
    try:
        yield set_copilot_call_count  # Yield the function itself
    finally:
        await set_copilot_call_count(access_token=actual_valid_access_token, count=0)


@pytest.fixture
def actual_user_file_service(
    mock_headers_with_actual_valid_access_token: dict,
) -> UserFileService:
    user_file_service = UserFileService(
        base_url=OPENBB_PAYMENTS_BASE_URL or "",
        access_token=mock_headers_with_actual_valid_access_token[
            "Authorization"
        ].replace("Bearer ", ""),
        user_id=mock_headers_with_actual_valid_access_token["X-User-Id"],
    )
    return user_file_service


@pytest.fixture
def test_txt_hitchhikers_guide_data() -> bytes:
    return """
    The answer to the ultimate question of life, the universe, and everything is 42.

    Don't Panic.

    So long, and thanks for all the fish.

    Time is an illusion. Lunchtime doubly so.

    I love deadlines. I love the whooshing noise they make as they go by.
    """.encode()


@pytest.fixture
def test_txt_amzn_data() -> bytes:
    with open(Path(__file__).parent / "test_data" / "amzn_data.txt", "rb") as f:
        content = f.read()
    return content


@pytest.fixture
def test_pdf_openbb_story_data() -> bytes:
    with open(Path(__file__).parent / "test_data" / "openbb_story.pdf", "rb") as f:
        content = f.read()
    return content


@pytest.fixture
def test_pdf_tsla_10q_data() -> bytes:
    with open(Path(__file__).parent / "test_data" / "tsla_10q_20240331.pdf", "rb") as f:
        content = f.read()
    return content


@pytest.fixture
def test_csv_tsla_historical_data() -> bytes:
    with open(Path(__file__).parent / "test_data" / "tsla_historical.csv", "rb") as f:
        content = f.read()
    return content


@pytest.fixture
def test_xlsx_management_comp_data() -> bytes:
    with open(
        Path(__file__).parent / "test_data" / "management_team_comp.xlsx", "rb"
    ) as f:
        content = f.read()
    return content


@pytest.fixture
def test_png_table_image_data() -> bytes:
    with open(Path(__file__).parent / "test_data" / "table.png", "rb") as f:
        content = f.read()
    return content


@pytest.fixture
def test_jpg_table_image_data() -> bytes:
    with open(Path(__file__).parent / "test_data" / "table.jpg", "rb") as f:
        content = f.read()
    return content


@pytest.fixture
def test_jpeg_table_image_data() -> bytes:
    with open(Path(__file__).parent / "test_data" / "table.jpeg", "rb") as f:
        content = f.read()
    return content


@pytest.fixture
def test_jpg_lion_image_data() -> bytes:
    with open(Path(__file__).parent / "test_data" / "animal_1.jpg", "rb") as f:
        content = f.read()
    return content


@pytest.fixture
def test_jpg_whale_image_data() -> bytes:
    with open(Path(__file__).parent / "test_data" / "animal_2.jpg", "rb") as f:
        content = f.read()
    return content


@pytest.fixture
def test_docx_tesla_wikipedia_data() -> bytes:
    with open(Path(__file__).parent / "test_data" / "tesla_wikipedia.docx", "rb") as f:
        content = f.read()
    return content


@pytest.fixture
def test_txt_hitchhikers_guide_user_file() -> UserFile:
    return UserFile(
        file_uuid=UUID("1fc69f4b-1fbc-472a-852a-cc96adff0701"),
        filename="hitchhikers_guide.txt",
        extension="txt",
    )


@pytest.fixture
def test_txt_amzn_data_user_file() -> UserFile:
    return UserFile(
        file_uuid=UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479"),
        filename="amzn_data.txt",
        extension="txt",
    )


@pytest.fixture
def test_pdf_openbb_story_user_file() -> UserFile:
    return UserFile(
        file_uuid=UUID("08d47a2f-bd35-4f53-a0e6-a45b4c7252f0"),
        filename="openbb_story.pdf",
        extension="pdf",
    )


@pytest.fixture
def test_pdf_tsla_10q_user_file() -> UserFile:
    return UserFile(
        file_uuid=UUID("5815d0c5-e83a-4098-8f66-707e874c891b"),
        filename="tsla_10q.pdf",
        extension="pdf",
    )


@pytest.fixture
def test_csv_tsla_historical_user_file() -> UserFile:
    return UserFile(
        file_uuid=UUID("e47ac10b-58cc-4372-a567-0e02b2c3d479"),
        filename="tsla_historical.csv",
        extension="csv",
    )


@pytest.fixture
def test_xlsx_management_comp_user_file() -> UserFile:
    return UserFile(
        file_uuid=UUID("ea09cd3a-cc6a-4b02-8032-7bf0b542a0c8"),
        filename="management_team_comp.xlsx",
        extension="xlsx",
    )


@pytest.fixture
def test_png_table_image_user_file() -> UserFile:
    return UserFile(
        file_uuid=UUID("266f83d5-c6c2-462c-9eee-7f08cff197e4"),
        filename="table.png",
        extension="png",
    )


@pytest.fixture
def test_jpg_table_image_user_file() -> UserFile:
    return UserFile(
        file_uuid=UUID("a87ecde4-c296-40fc-aaf8-a2b6e2daffdb"),
        filename="table.jpg",
        extension="jpg",
    )


@pytest.fixture
def test_jpeg_table_image_user_file() -> UserFile:
    return UserFile(
        file_uuid=UUID("7b9a773f-bed7-495f-97c1-e026e10eaf93"),
        filename="table.jpeg",
        extension="jpeg",
    )


@pytest.fixture
def test_jpg_lion_image_user_file() -> UserFile:
    return UserFile(
        file_uuid=UUID("11242bf8-02c3-497f-830e-957c8bf4fdb3"),
        filename="animal_1.jpg",
        extension="jpg",
    )


@pytest.fixture
def test_jpg_whale_image_user_file() -> UserFile:
    return UserFile(
        file_uuid=UUID("f49b59ff-ec8b-4437-8367-f1d5948c2750"),
        filename="animal_2.jpg",
        extension="jpg",
    )


@pytest.fixture
def test_docx_tesla_wikipedia_user_file() -> UserFile:
    return UserFile(
        file_uuid=UUID("3f26673e-bb28-488a-bb0c-c3a75229eed4"),
        filename="tesla_wikipedia.docx",
        extension="docx",
    )


@pytest.fixture
def test_txt_hitchhikers_guide_downloaded_user_file(
    test_txt_hitchhikers_guide_user_file: UserFile,
    test_txt_hitchhikers_guide_data: bytes,
    mock_uuids: type[MockUUIDs],
) -> Document:
    return Document(
        file_uuid=test_txt_hitchhikers_guide_user_file.file_uuid,
        filename=test_txt_hitchhikers_guide_user_file.filename,
        extension=test_txt_hitchhikers_guide_user_file.extension,
        content=test_txt_hitchhikers_guide_data,
        source_info=SourceInfo(
            type="widget",
            uuid=UUID(mock_uuids.ID1.value),
            name=test_txt_hitchhikers_guide_user_file.filename,
        ),
    )


@pytest.fixture
def test_txt_amzn_data_downloaded_user_file(
    test_txt_amzn_data_user_file: UserFile,
    test_txt_amzn_data: bytes,
    mock_uuids: type[MockUUIDs],
) -> Document:
    return Document(
        file_uuid=test_txt_amzn_data_user_file.file_uuid,
        filename=test_txt_amzn_data_user_file.filename,
        extension=test_txt_amzn_data_user_file.extension,
        content=test_txt_amzn_data,
        source_info=SourceInfo(
            type="widget",
            uuid=UUID(mock_uuids.ID2.value),
            name=test_txt_amzn_data_user_file.filename,
        ),
    )


@pytest.fixture
def test_pdf_openbb_story_downloaded_user_file(
    test_pdf_openbb_story_user_file: UserFile,
    test_pdf_openbb_story_data: bytes,
    mock_uuids: type[MockUUIDs],
) -> Document:
    return Document(
        file_uuid=test_pdf_openbb_story_user_file.file_uuid,
        filename=test_pdf_openbb_story_user_file.filename,
        extension=test_pdf_openbb_story_user_file.extension,
        content=test_pdf_openbb_story_data,
        source_info=SourceInfo(
            type="widget",
            uuid=UUID(mock_uuids.ID3.value),
            name=test_pdf_openbb_story_user_file.filename,
        ),
    )


@pytest.fixture
def test_pdf_tsla_10q_downloaded_user_file(
    test_pdf_tsla_10q_user_file: UserFile,
    test_pdf_tsla_10q_data: bytes,
    mock_uuids: type[MockUUIDs],
) -> Document:
    return Document(
        file_uuid=test_pdf_tsla_10q_user_file.file_uuid,
        filename=test_pdf_tsla_10q_user_file.filename,
        extension=test_pdf_tsla_10q_user_file.extension,
        content=test_pdf_tsla_10q_data,
        source_info=SourceInfo(
            type="widget",
            uuid=UUID(mock_uuids.ID4.value),
            name=test_pdf_tsla_10q_user_file.filename,
        ),
    )


@pytest.fixture
def test_csv_tsla_historical_downloaded_user_file(
    test_csv_tsla_historical_user_file: UserFile,
    test_csv_tsla_historical_data: bytes,
    mock_uuids: type[MockUUIDs],
) -> Document:
    return Document(
        file_uuid=test_csv_tsla_historical_user_file.file_uuid,
        filename=test_csv_tsla_historical_user_file.filename,
        extension=test_csv_tsla_historical_user_file.extension,
        content=test_csv_tsla_historical_data,
        source_info=SourceInfo(
            type="widget",
            uuid=mock_uuids.ID5.value,
            name=test_csv_tsla_historical_user_file.filename,
            metadata={
                "filename": test_csv_tsla_historical_user_file.filename,
                "extension": test_csv_tsla_historical_user_file.extension,
            },
        ),
    )


@pytest.fixture
def test_xlsx_management_comp_downloaded_user_file(
    test_xlsx_management_comp_user_file: UserFile,
    test_xlsx_management_comp_data: bytes,
    mock_uuids: type[MockUUIDs],
) -> Document:
    return Document(
        file_uuid=test_xlsx_management_comp_user_file.file_uuid,
        filename=test_xlsx_management_comp_user_file.filename,
        extension=test_xlsx_management_comp_user_file.extension,
        content=test_xlsx_management_comp_data,
        source_info=SourceInfo(
            type="widget",
            uuid=mock_uuids.ID6.value,
            name=test_xlsx_management_comp_user_file.filename,
        ),
    )


@pytest.fixture
def test_png_table_image_downloaded_user_file(
    test_png_table_image_user_file: UserFile,
    test_png_table_image_data: bytes,
    mock_uuids: type[MockUUIDs],
) -> Document:
    return Document(
        file_uuid=test_png_table_image_user_file.file_uuid,
        filename=test_png_table_image_user_file.filename,
        extension=test_png_table_image_user_file.extension,
        content=test_png_table_image_data,
        source_info=SourceInfo(
            type="widget",
            uuid=mock_uuids.ID7.value,
            name=test_png_table_image_user_file.filename,
        ),
    )


@pytest.fixture
def test_jpg_table_image_downloaded_user_file(
    test_jpg_table_image_user_file: UserFile,
    test_jpg_table_image_data: bytes,
    mock_uuids: type[MockUUIDs],
) -> Document:
    return Document(
        file_uuid=test_jpg_table_image_user_file.file_uuid,
        filename=test_jpg_table_image_user_file.filename,
        extension=test_jpg_table_image_user_file.extension,
        content=test_jpg_table_image_data,
        source_info=SourceInfo(
            type="widget",
            uuid=mock_uuids.ID8.value,
            name=test_jpg_table_image_user_file.filename,
        ),
    )


@pytest.fixture
def test_jpeg_table_image_downloaded_user_file(
    test_jpeg_table_image_user_file: UserFile,
    test_jpeg_table_image_data: bytes,
    mock_uuids: type[MockUUIDs],
) -> Document:
    return Document(
        file_uuid=test_jpeg_table_image_user_file.file_uuid,
        filename=test_jpeg_table_image_user_file.filename,
        extension=test_jpeg_table_image_user_file.extension,
        content=test_jpeg_table_image_data,
        source_info=SourceInfo(
            type="widget",
            uuid=mock_uuids.ID9.value,
            name=test_jpeg_table_image_user_file.filename,
        ),
    )


@pytest.fixture
def test_jpg_lion_image_downloaded_user_file(
    test_jpg_lion_image_user_file: UserFile,
    test_jpg_lion_image_data: bytes,
    mock_uuids: type[MockUUIDs],
) -> Document:
    return Document(
        file_uuid=test_jpg_lion_image_user_file.file_uuid,
        filename=test_jpg_lion_image_user_file.filename,
        extension=test_jpg_lion_image_user_file.extension,
        content=test_jpg_lion_image_data,
        source_info=SourceInfo(
            type="widget",
            uuid=mock_uuids.ID10.value,
            name=test_jpg_lion_image_user_file.filename,
        ),
    )


@pytest.fixture
def test_jpg_whale_image_downloaded_user_file(
    test_jpg_whale_image_user_file: UserFile,
    test_jpg_whale_image_data: bytes,
    mock_uuids: type[MockUUIDs],
) -> Document:
    return Document(
        file_uuid=test_jpg_whale_image_user_file.file_uuid,
        filename=test_jpg_whale_image_user_file.filename,
        extension=test_jpg_whale_image_user_file.extension,
        content=test_jpg_whale_image_data,
        source_info=SourceInfo(
            type="widget",
            uuid=mock_uuids.ID11.value,
            name=test_jpg_whale_image_user_file.filename,
        ),
    )


@pytest.fixture
def test_docx_tesla_wikipedia_downloaded_user_file(
    test_docx_tesla_wikipedia_user_file: UserFile,
    test_docx_tesla_wikipedia_data: bytes,
    mock_uuids: type[MockUUIDs],
) -> Document:
    return Document(
        file_uuid=test_docx_tesla_wikipedia_user_file.file_uuid,
        filename=test_docx_tesla_wikipedia_user_file.filename,
        extension=test_docx_tesla_wikipedia_user_file.extension,
        content=test_docx_tesla_wikipedia_data,
        source_info=SourceInfo(
            type="widget",
            uuid=mock_uuids.ID12.value,
            name=test_docx_tesla_wikipedia_user_file.filename,
        ),
    )


@pytest_asyncio.fixture
async def actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service(
    actual_user_file_service: UserFileService,
    test_txt_hitchhikers_guide_user_file: UserFile,
    test_txt_hitchhikers_guide_data: bytes,
) -> UserFile:
    return await actual_user_file_service.upload_file(
        filename=test_txt_hitchhikers_guide_user_file.filename,
        content=test_txt_hitchhikers_guide_data,
    )


@pytest_asyncio.fixture
async def actual_txt_amzn_data_user_file_stored_on_user_file_service(
    actual_user_file_service: UserFileService,
    test_txt_amzn_data_user_file: UserFile,
    test_txt_amzn_data: bytes,
) -> UserFile:
    return await actual_user_file_service.upload_file(
        filename=test_txt_amzn_data_user_file.filename,
        content=test_txt_amzn_data,
    )


@pytest_asyncio.fixture
async def actual_pdf_openbb_story_user_file_stored_on_user_file_service(
    actual_user_file_service: UserFileService,
    test_pdf_openbb_story_user_file: UserFile,
    test_pdf_openbb_story_data: bytes,
) -> UserFile:
    return await actual_user_file_service.upload_file(
        filename=test_pdf_openbb_story_user_file.filename,
        content=test_pdf_openbb_story_data,
    )


@pytest_asyncio.fixture
async def actual_csv_tsla_historical_user_file_stored_on_user_file_service(
    actual_user_file_service: UserFileService, test_csv_tsla_historical_data: bytes
) -> UserFile:
    return await actual_user_file_service.upload_file(
        filename="tsla_historical.csv", content=test_csv_tsla_historical_data
    )


@pytest_asyncio.fixture
async def actual_docx_tesla_wikipedia_user_file_stored_on_user_file_service(
    actual_user_file_service: UserFileService, test_docx_tesla_wikipedia_data: bytes
) -> UserFile:
    return await actual_user_file_service.upload_file(
        filename="tesla_wikipedia.docx", content=test_docx_tesla_wikipedia_data
    )


@pytest_asyncio.fixture
async def actual_png_table_image_user_file_stored_on_user_file_service(
    actual_user_file_service: UserFileService, test_png_table_image_data: bytes
) -> UserFile:
    return await actual_user_file_service.upload_file(
        filename="table.png", content=test_png_table_image_data
    )


@pytest_asyncio.fixture
async def actual_txt_hitchhikers_guide_associated_vector_db_stored_on_user_file_service(  # noqa: E501
    actual_user_file_service: UserFileService,
    actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service: UserFile,
    mock_uuids: type[MockUUIDs],
) -> UserFile:
    documents = [
        VectorDbDocument(
            page_content="The answer to the ultimate question of life, the universe, and everything is 42.",  # noqa: E501
            metadata={
                "name": "hitchhikers_guide.txt",
                "page_number": 1,
                "document_hash": xxhash.xxh64_hexdigest("content-1"),
            },
        ),
        VectorDbDocument(
            page_content="Don't Panic.",
            metadata={
                "name": "hitchhikers_guide.txt",
                "page_number": 2,
                "document_hash": xxhash.xxh64_hexdigest("content-2"),
            },
        ),
        VectorDbDocument(
            page_content="So long, and thanks for all the fish.",
            metadata={
                "name": "hitchhikers_guide.txt",
                "page_number": 3,
                "document_hash": xxhash.xxh64_hexdigest("content-3"),
            },
        ),
        VectorDbDocument(
            page_content="Time is an illusion. Lunchtime doubly so.",
            metadata={
                "name": "hitchhikers_guide.txt",
                "page_number": 4,
                "document_hash": xxhash.xxh64_hexdigest("content-4"),
            },
        ),
        VectorDbDocument(
            page_content="I love deadlines. I love the whooshing noise they make as they go by.",  # noqa: E501
            metadata={
                "name": "hitchhikers_guide.txt",
                "page_number": 5,
                "document_hash": xxhash.xxh64_hexdigest("content-5"),
            },
        ),
    ]
    vector_db = VectorDb()
    await vector_db.add(documents)
    buffer = io.BytesIO()
    vector_db.save_to_buffer(buffer)
    return await actual_user_file_service.upload_file(
        filename=f"{actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.file_uuid}-vectordb.zip",  # noqa: E501
        content=buffer.getvalue(),
    )


@pytest_asyncio.fixture
async def actual_pdf_openbb_story_associated_vector_db_stored_on_user_file_service(
    actual_user_file_service: UserFileService,
    test_pdf_openbb_story_data: bytes,
    actual_pdf_openbb_story_user_file_stored_on_user_file_service: UserFile,
    mock_uuids: type[MockUUIDs],
) -> UserFile:
    vector_db = VectorDb()
    await vector_db.add(
        documents=[
            VectorDbDocument(
                page_content="OpenBB is a fully remote company!",
                metadata={
                    "name": actual_pdf_openbb_story_user_file_stored_on_user_file_service.filename,  # noqa: E501
                    "page_number": 1,
                    "document_hash": xxhash.xxh64_hexdigest(test_pdf_openbb_story_data),
                },
            )
        ]
    )
    buffer = io.BytesIO()
    vector_db.save_to_buffer(buffer)
    return await actual_user_file_service.upload_file(
        filename=f"{actual_pdf_openbb_story_user_file_stored_on_user_file_service.file_uuid}-vectordb.zip",  # noqa: E501
        content=buffer.getvalue(),
    )


@pytest.fixture
def test_client(
    test_user_file_service: UserFileService,
    test_sql_agent_service: SqlAgentService,
):
    app.dependency_overrides[get_user_file_service] = lambda: test_user_file_service
    app.dependency_overrides[get_sql_agent_service] = lambda: test_sql_agent_service
    client = TestClient(app)
    yield client

    # Clean up overrides after test
    app.dependency_overrides.pop(get_user_file_service, None)
    app.dependency_overrides.pop(get_sql_agent_service, None)


@pytest.fixture
def no_rate_limit() -> None:
    with patch("openbb_ada.utils.utils.is_rate_limited") as mock_is_rate_limited:
        with patch(
            "openbb_ada.utils.utils.increment_copilot_call_count"
        ) as mock_increment_copilot_call_count:
            mock_is_rate_limited.return_value = False
            mock_increment_copilot_call_count.return_value = None
            yield


@pytest.fixture
def mock_request():
    yield Mock()


@pytest.fixture
def mock_200_response():
    mock_response = Mock()
    mock_response.status_code = 200
    yield mock_response


@pytest.fixture
def mock_500_response():
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Something went wrong", request=Mock(), response=Mock()
    )
    yield mock_response


@pytest.fixture
def test_template_service() -> TemplateService:
    return TemplateService()


@pytest.fixture
def test_user_file_service(
    test_txt_hitchhikers_guide_user_file: UserFile,
    test_txt_amzn_data_user_file: UserFile,
    test_csv_tsla_historical_user_file: UserFile,
    test_xlsx_management_comp_user_file: UserFile,
    test_pdf_openbb_story_user_file: UserFile,
    test_pdf_tsla_10q_user_file: UserFile,
    test_png_table_image_user_file: UserFile,
    test_jpg_table_image_user_file: UserFile,
    test_jpeg_table_image_user_file: UserFile,
    test_docx_tesla_wikipedia_user_file: UserFile,
    test_txt_hitchhikers_guide_downloaded_user_file: Document,
    test_txt_amzn_data_downloaded_user_file: Document,
    test_csv_tsla_historical_downloaded_user_file: Document,
    test_xlsx_management_comp_downloaded_user_file: Document,
    test_pdf_openbb_story_downloaded_user_file: Document,
    test_pdf_tsla_10q_downloaded_user_file: Document,
    test_png_table_image_downloaded_user_file: Document,
    test_jpg_table_image_downloaded_user_file: Document,
    test_jpeg_table_image_downloaded_user_file: Document,
    test_docx_tesla_wikipedia_downloaded_user_file: Document,
    mock_headers: dict,
) -> Generator[UserFileService, None, None]:
    with patch(
        "openbb_ada.services.UserFileService.bulk_upload_files",
        return_value=AsyncMock(),
    ):
        with patch(
            "openbb_ada.services.UserFileService.bulk_download_files",
            new_callable=AsyncMock,
        ) as mock_bulk_download_files:
            file_contents = {
                test_txt_hitchhikers_guide_user_file.file_uuid: test_txt_hitchhikers_guide_downloaded_user_file.content,  # noqa: E501
                test_txt_amzn_data_user_file.file_uuid: test_txt_amzn_data_downloaded_user_file.content,  # noqa: E501
                test_csv_tsla_historical_user_file.file_uuid: test_csv_tsla_historical_downloaded_user_file.content,  # noqa: E501
                test_xlsx_management_comp_user_file.file_uuid: test_xlsx_management_comp_downloaded_user_file.content,  # noqa: E501
                test_pdf_openbb_story_user_file.file_uuid: test_pdf_openbb_story_downloaded_user_file.content,  # noqa: E501
                test_pdf_tsla_10q_user_file.file_uuid: test_pdf_tsla_10q_downloaded_user_file.content,  # noqa: E501
                test_png_table_image_user_file.file_uuid: test_png_table_image_downloaded_user_file.content,  # noqa: E501
                test_jpg_table_image_user_file.file_uuid: test_jpg_table_image_downloaded_user_file.content,  # noqa: E501
                test_jpeg_table_image_user_file.file_uuid: test_jpeg_table_image_downloaded_user_file.content,  # noqa: E501
                test_docx_tesla_wikipedia_user_file.file_uuid: test_docx_tesla_wikipedia_downloaded_user_file.content,  # noqa: E501
            }

        async def side_effect(
            target_files: list[UserFile],
        ) -> list[Document]:
            downloaded_files = []
            for target_file in target_files:
                if target_file.file_uuid in file_contents:
                    downloaded_files.append(
                        Document(
                            file_uuid=target_file.file_uuid,
                            filename=target_file.filename,
                            extension=target_file.extension,
                            content=file_contents[target_file.file_uuid],
                            source_info=target_file.source_info,
                        )
                    )
            return downloaded_files

        mock_bulk_download_files.side_effect = side_effect

        with patch(
            "openbb_ada.services.UserFileService.bulk_download_external_files",
            return_value=AsyncMock(),
        ) as mock_download_external_files:
            # Mapping file names to their configuration details
            documents_mapping = {
                "openbb_story.pdf": {
                    "doc": test_pdf_openbb_story_downloaded_user_file,
                    "extension": "pdf",
                    "widget_id": "openbb_story",
                    "name": "OpenBB Story",
                    "description": "Tells the story of OpenBB.",
                    "metadata": {
                        "filename": "openbb_story.pdf",
                        "file_extension": "pdf",
                    },
                },
                "tsla_10q.pdf": {
                    "doc": test_pdf_tsla_10q_downloaded_user_file,
                    "extension": "pdf",
                    "widget_id": "tsla_10q",
                    "name": "TSLA 10Q",
                    "description": "TSLA 10Q data.",
                    "metadata": {
                        "filename": "tsla_10q.pdf",
                        "file_extension": "pdf",
                    },
                },
                "table_image.png": {
                    "doc": test_png_table_image_downloaded_user_file,
                    "extension": "png",
                    "widget_id": "table_image",
                    "name": "Table Image",
                    "description": "Table image data.",
                    "metadata": {
                        "filename": "table_image.png",
                        "file_extension": "png",
                    },
                },
                "table_image.jpg": {
                    "doc": test_jpg_table_image_downloaded_user_file,
                    "extension": "jpg",
                    "widget_id": "table_image",
                    "name": "Table Image",
                    "description": "Table image data.",
                    "metadata": {
                        "filename": "table_image.jpg",
                        "file_extension": "jpg",
                    },
                },
                "table_image.jpeg": {
                    "doc": test_jpeg_table_image_downloaded_user_file,
                    "extension": "jpeg",
                    "widget_id": "table_image",
                    "name": "Table Image",
                    "description": "Table image data.",
                    "metadata": {
                        "filename": "table_image.jpg",
                        "file_extension": "jpg",
                    },
                },
                "tsla_historical.csv": {
                    "doc": test_csv_tsla_historical_downloaded_user_file,
                    "extension": "csv",
                    "widget_id": "tsla_historical",
                    "name": "TSLA Historical",
                    "description": "TSLA's historical data.",
                    "metadata": {
                        "filename": "tsla_historical.csv",
                        "file_extension": "csv",
                    },
                },
                "management_team_comp.xlsx": {
                    "doc": test_xlsx_management_comp_downloaded_user_file,
                    "extension": "xlsx",
                    "widget_id": "management_comp",
                    "name": "Management Comp",
                    "description": "Management compensation data.",
                    "metadata": {
                        "filename": "management_team_comp.xlsx",
                        "file_extension": "xlsx",
                    },
                },
                "hitchhikers_guide.txt": {
                    "doc": test_txt_hitchhikers_guide_downloaded_user_file,
                    "extension": "txt",
                    "widget_id": "hitchhikers_guide",
                    "name": "Hitchhiker's Guide",
                    "description": "The Hitchhiker's Guide to the Galaxy.",
                    "metadata": {
                        "filename": "hitchhikers_guide.txt",
                        "file_extension": "txt",
                    },
                },
                "tesla_wikipedia.docx": {
                    "doc": test_docx_tesla_wikipedia_downloaded_user_file,
                    "extension": "docx",
                    "widget_id": "tesla_wikipedia",
                    "name": "Tesla Wikipedia",
                    "description": "Tesla Wikipedia data.",
                    "metadata": {
                        "filename": "tesla_wikipedia.docx",
                        "file_extension": "docx",
                    },
                },
            }

            async def side_effect_download_external_files(
                url_file_references: list[UrlFileReference],
            ) -> list[Document | UnavailableDocument]:
                documents: list[Document | UnavailableDocument] = []
                for url_file_reference in url_file_references:
                    config = documents_mapping.get(url_file_reference.filename)
                    if config:
                        doc_obj = config["doc"]
                        documents.append(
                            Document(
                                content=doc_obj.content,
                                filename=url_file_reference.filename,
                                extension=url_file_reference.extension,
                                source_info=SourceInfo(
                                    uuid=url_file_reference.source_info.uuid,
                                    origin=url_file_reference.source_info.origin,
                                    widget_id=url_file_reference.source_info.widget_id,
                                    type=url_file_reference.source_info.type,
                                    name=url_file_reference.source_info.name,
                                    description=url_file_reference.source_info.description,
                                    metadata=url_file_reference.source_info.metadata,
                                ),
                            )
                        )
                    else:
                        documents.append(
                            UnavailableDocument(
                                error=f"Failed to download file: {url_file_reference.url}, error: Link expired.",  # noqa: E501
                                source_info=url_file_reference.source_info,
                            )
                        )

                return documents

            mock_download_external_files.side_effect = (
                side_effect_download_external_files
            )
            file_service = UserFileService(
                base_url="https://mock-hub-api-backend.url",
                access_token=mock_headers["Authorization"].replace("Bearer ", ""),
                user_id=mock_headers["X-User-Id"],
            )
            yield file_service


@pytest.fixture
def test_sql_agent_service(
    test_template_service: TemplateService,
    test_database_engine,
) -> SqlAgentService:
    return SqlAgentService(
        template_service=test_template_service,
        logging_service=Mock(),
        openai_api_key=None,
        engine=test_database_engine,
    )


@pytest.fixture
def test_document_agent_service(
    test_template_service: TemplateService,
) -> DocumentAgentService:
    return DocumentAgentService(
        template_service=test_template_service,
        logging_service=LoggingService(),
        openai_api_key=None,
    )  # Replace logging service with mock


@pytest.fixture
def test_document_service(
    test_user_file_service: UserFileService,
    test_template_service: TemplateService,
    test_sql_agent_service: SqlAgentService,
    test_document_agent_service: DocumentAgentService,
    mock_headers: str,
) -> DocumentService:
    document_service = DocumentService(
        logging_service=LoggingService(),
        user_file_service=test_user_file_service,
        template_service=test_template_service,
        sql_agent_service=test_sql_agent_service,
        document_agent_service=test_document_agent_service,
        documents=[],
        openai_api_key=None,
    )
    return document_service


@pytest.fixture
def test_sql_query_generation_service(
    test_template_service: TemplateService,
) -> SqlQueryGenerationService:
    return SqlQueryGenerationService(
        template_service=test_template_service,
        logging_service=LoggingService(),
        openai_api_key=None,
    )


@pytest.fixture
def test_copilot_service(
    test_document_service: DocumentService,
    test_template_service: TemplateService,
    test_sql_query_generation_service: SqlQueryGenerationService,
) -> CopilotService:
    return CopilotService(
        user_id="test-user-id",
        document_service=test_document_service,
        context_service=Mock(),
        template_service=test_template_service,
        url_retrieval_service=Mock(),
        copilot_data_service=Mock(),
        logging_service=Mock(),
        client_function_call_service=ClientFunctionCallService(
            copilot_data_service=Mock(),
            logging_service=LoggingService(),
        ),
        native_function_call_service=NativeFunctionCallService(
            document_service=test_document_service,
            logging_service=LoggingService(),
            web_search_llm_service=Mock(),
            context_service=Mock(),
            template_service=test_template_service,
            citation_service=Mock(),
            sql_query_generation_service=test_sql_query_generation_service,
        ),
        citation_service=Mock(),
        mcp_data_service=Mock(),
        openai_api_key=None,
        sql_query_generation_service=test_sql_query_generation_service,
    )


@pytest.fixture()
def mock_httpx_async_client():
    with patch(
        "openbb_ada.services.httpx.AsyncClient",
        return_value=Mock(name="httpx context manager"),
    ) as mock_context_manager:
        mock_instance = AsyncMock(name="httpx context manager instance")
        mock_client = AsyncMock(name="httpx client")

        mock_context_manager.return_value = mock_instance
        mock_instance.__aenter__.return_value = mock_client

        yield mock_client
