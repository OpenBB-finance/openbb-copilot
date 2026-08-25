import uuid
from pathlib import Path

import pytest

from openbb_ada.models import (
    Document,
    HttpMethod,
    SourceInfo,
    TargetUserFile,
    UrlFileReference,
    UserFile,
)
from openbb_ada.services import UserFileService
from tests.conftest import MockUUIDs


@pytest.mark.skip(reason="See TODO in service code. Not used and pending deletion.")
@pytest.mark.asyncio
@pytest.mark.integration
async def test_user_file_service_upload_file(actual_user_file_service: UserFileService):
    actual_result = await actual_user_file_service.upload_file(
        "test.txt", "test content"
    )
    assert isinstance(actual_result, UserFile)
    assert actual_result.file_uuid is not None
    assert actual_result.filename == "test.txt"
    assert actual_result.extension == "txt"


@pytest.mark.skip(reason="See TODO in service code. Not used and pending deletion.")
@pytest.mark.asyncio
@pytest.mark.integration
async def test_user_file_service_bulk_upload_files(
    actual_user_file_service: UserFileService,
):
    test_files = [
        ("test1.txt", b"test content 1"),
        ("test2.txt", b"test content 2"),
    ]
    actual_result = await actual_user_file_service.bulk_upload_files(test_files)
    assert len(actual_result) == 2
    assert all(isinstance(file, UserFile) for file in actual_result)

    assert all(file.filename in ("test1.txt", "test2.txt") for file in actual_result)
    assert all(file.extension in ("txt", "txt") for file in actual_result)


@pytest.mark.skip(reason="See TODO in service code. Not used and pending deletion.")
@pytest.mark.asyncio
@pytest.mark.integration
async def test_user_file_service_read_file(actual_user_file_service: UserFileService):
    # No real way of testing reading files in isolation against the real
    # backend. So we implicitly depend on the upload working.
    test_file_path = Path(__file__).parent / "test_data" / "tsla_historical.csv"
    uploaded_file = await actual_user_file_service.upload_file(
        "tsla_historical.csv", test_file_path.read_text()
    )
    actual_result = await actual_user_file_service.read_file(uploaded_file)
    assert isinstance(actual_result, Document)
    assert actual_result.content.decode("utf-8") == test_file_path.read_text()


@pytest.mark.skip(reason="See TODO in service code. Not used and pending deletion.")
@pytest.mark.asyncio
@pytest.mark.integration
async def test_user_file_service_bulk_download_files(
    actual_user_file_service: UserFileService,
    test_pdf_openbb_story_data: bytes,
    test_csv_tsla_historical_data: bytes,
    actual_pdf_openbb_story_user_file_stored_on_user_file_service: UserFile,
    actual_csv_tsla_historical_user_file_stored_on_user_file_service: UserFile,
):
    actual_result = await actual_user_file_service.bulk_download_files(
        [
            TargetUserFile(
                **actual_pdf_openbb_story_user_file_stored_on_user_file_service.model_dump(),
                source_info=SourceInfo(
                    type="widget",
                    uuid=actual_pdf_openbb_story_user_file_stored_on_user_file_service.file_uuid,
                    name=actual_pdf_openbb_story_user_file_stored_on_user_file_service.filename,
                ),
            ),
            TargetUserFile(
                **actual_csv_tsla_historical_user_file_stored_on_user_file_service.model_dump(),
                source_info=SourceInfo(
                    type="widget",
                    uuid=actual_csv_tsla_historical_user_file_stored_on_user_file_service.file_uuid,
                    name=actual_csv_tsla_historical_user_file_stored_on_user_file_service.filename,
                ),
            ),
        ]
    )
    assert len(actual_result) == 2
    assert all(isinstance(file, Document) for file in actual_result)
    assert all(file.content is not None for file in actual_result)
    assert all(
        file.content == test_pdf_openbb_story_data
        or file.content == test_csv_tsla_historical_data
        for file in actual_result
    )


@pytest.mark.skip(reason="See TODO in service code. Not used and pending deletion.")
@pytest.mark.asyncio
@pytest.mark.integration
async def test_user_file_service_bulk_download_files_single_file(
    actual_user_file_service: UserFileService,
    test_pdf_openbb_story_data: bytes,
    actual_pdf_openbb_story_user_file_stored_on_user_file_service: UserFile,
):
    actual_result = await actual_user_file_service.bulk_download_files(
        [
            TargetUserFile(
                **actual_pdf_openbb_story_user_file_stored_on_user_file_service.model_dump(),
                source_info=SourceInfo(
                    type="widget",
                    uuid=actual_pdf_openbb_story_user_file_stored_on_user_file_service.file_uuid,
                    name=actual_pdf_openbb_story_user_file_stored_on_user_file_service.filename,
                ),
            ),
        ]
    )
    assert len(actual_result) == 1
    assert isinstance(actual_result[0], Document)
    assert actual_result[0].content == test_pdf_openbb_story_data


@pytest.mark.skip(reason="See TODO in service code. Not used and pending deletion.")
@pytest.mark.asyncio
@pytest.mark.integration
async def test_user_file_service_bulk_download_external_files(
    actual_user_file_service: UserFileService,
    test_pdf_openbb_story_data: bytes,
    mock_uuids: type[MockUUIDs],
):
    test_source_info = SourceInfo(
        type="widget",
        uuid=uuid.UUID(mock_uuids.ID1.value),
        name="Sample PDF",
        description="This is a sample PDF",
        origin="test-origin",
        widget_id="test-widget-id",
    )

    actual_result = await actual_user_file_service.bulk_download_external_files(
        [
            UrlFileReference(
                url="https://openbb-assets.s3.us-east-1.amazonaws.com/testing/openbb_story.pdf",
                filename="sample.pdf",
                extension="pdf",
                source_info=test_source_info,
            )
        ]
    )

    assert len(actual_result) == 1
    assert isinstance(actual_result[0], Document)
    assert actual_result[0].content == test_pdf_openbb_story_data
    assert actual_result[0].filename == "sample.pdf"
    assert actual_result[0].extension == "pdf"
    assert actual_result[0].source_info == test_source_info


@pytest.mark.skip(reason="See TODO in service code. Not used and pending deletion.")
@pytest.mark.asyncio
@pytest.mark.integration
async def test_user_file_service_bulk_download_external_files_hub_presigned_url(
    mock_uuids: type[MockUUIDs],
    actual_user_file_service: UserFileService,
    test_pdf_openbb_story_data: bytes,
):
    upload_response = await actual_user_file_service.upload_file(
        filename="unit-test.pdf",
        content=test_pdf_openbb_story_data,
    )
    file_name = str(upload_response.file_uuid) + "." + upload_response.extension
    # Now let's create a pre-signed URL for this file.
    presigned_url_response = await actual_user_file_service._make_request(
        method=HttpMethod.GET,
        path=f"/pro/files/{file_name}/presigned-url",
    )
    presigned_url_response = presigned_url_response.json()

    test_source_info = SourceInfo(
        type="widget",
        uuid=uuid.UUID(mock_uuids.ID1.value),
        name="Sample PDF",
        description="This is a sample PDF",
        origin="test-origin",
        widget_id="test-widget-id",
    )

    actual_result = await actual_user_file_service.bulk_download_external_files(
        [
            UrlFileReference(
                url=presigned_url_response["pre_signed_url"],
                filename=presigned_url_response["original_file_name"],
                extension=presigned_url_response["original_file_name"].split(".")[-1],
                source_info=test_source_info,
            )
        ]
    )

    assert len(actual_result) == 1
    assert actual_result[0].filename == "unit-test.pdf"
    assert actual_result[0].extension == "pdf"
    assert actual_result[0].content == test_pdf_openbb_story_data
    assert actual_result[0].source_info == test_source_info
