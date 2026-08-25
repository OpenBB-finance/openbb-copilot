from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from openbb_ai.models import SourceInfo, StatusUpdateSSE, UserAPIKeys

from openbb_ada.dependencies import get_document_service
from openbb_ada.models import (
    Document,
    DocumentQueryResult,
    UserFile,
)
from openbb_ada.services import (
    DocumentAgentService,
    DocumentService,
    LoggingService,
    SqlAgentService,
    TemplateService,
    UserFileService,
)
from tests.conftest import MockUUIDs


@pytest.mark.skip(reason="Integration tests are skipped and will be rethought.")
@pytest.mark.asyncio
@pytest.mark.integration
async def test_document_service__init(
    test_txt_hitchhikers_guide_data: bytes,
    actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service: UserFile,
    actual_txt_hitchhikers_guide_associated_vector_db_stored_on_user_file_service: UserFile,  # noqa: E501
    actual_user_file_service: UserFileService,
    test_document_agent_service: DocumentAgentService,
    mock_uuids: type[MockUUIDs],
):
    document_service = await get_document_service(
        user_file_service=actual_user_file_service,
        template_service=Mock(),
        document_agent_service=test_document_agent_service,
        sql_agent_service=Mock(),
        logging_service=LoggingService(),
        documents=[
            Document(
                file_uuid=actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.file_uuid,
                content=test_txt_hitchhikers_guide_data,
                filename=actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.filename,
                extension=actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.extension,
                source_info=SourceInfo(
                    type="widget",
                    name=actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.filename,
                    uuid=UUID(mock_uuids.ID1.value),
                ),
            )
        ],
        api_keys=UserAPIKeys(openai_api_key="test_openai_api_key"),
    )

    assert len(document_service.downloaded_user_files) == 1
    assert len(document_service.documents) == 1

    document = document_service.documents[0]
    assert document.hash == "87612fe692e1c41b"
    assert (
        document.file_uuid
        == actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.file_uuid
    )
    assert (
        document.filename
        == actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.filename
    )
    assert (
        document.extension
        == actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.extension
    )

    assert document_service._openai_api_key == "test_openai_api_key"


@pytest.mark.skip(reason="Integration tests are skipped and will be rethought.")
@pytest.mark.asyncio
@pytest.mark.integration
async def test_document_service__init_with_no_vector_db_creates_vector_db(
    test_txt_hitchhikers_guide_data: bytes,
    actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service: UserFile,
    actual_user_file_service: UserFileService,
    test_document_agent_service: DocumentAgentService,
    mock_uuids: type[MockUUIDs],
):
    test_document = Document(
        file_uuid=actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.file_uuid,
        # We add some random bytes to the end of the file to ensure that the
        # hash changes, forcing a new vector database to be created and uploaded
        # to hub.
        content=test_txt_hitchhikers_guide_data + str(uuid4()).encode(),
        filename=actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.filename,
        extension=actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.extension,
        source_info=SourceInfo(
            type="widget",
            name=actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.filename,
            uuid=UUID(mock_uuids.ID1.value),
        ),
    )

    document_service = await get_document_service(
        user_file_service=actual_user_file_service,
        template_service=Mock(),
        document_agent_service=test_document_agent_service,
        sql_agent_service=Mock(),
        logging_service=LoggingService(),
        # An uploaded file that doesn't have an associated vector database
        documents=[test_document],
        api_keys=None,
    )

    assert len(document_service.loaded_unstructured_files) == 1
    assert any(
        doc.hash == test_document.hash
        for doc in document_service.loaded_unstructured_files
    )


@pytest.mark.skip(reason="Integration tests are skipped and will be rethought.")
@pytest.mark.asyncio
@pytest.mark.integration
async def test_document_service__init_csv_file(
    test_csv_tsla_historical_data: bytes,
    actual_csv_tsla_historical_user_file_stored_on_user_file_service: UserFile,
    actual_user_file_service: UserFileService,
    test_sql_agent_service: SqlAgentService,
    test_document_agent_service: DocumentAgentService,
    mock_uuids: type[MockUUIDs],
):
    document_service = await get_document_service(
        user_file_service=actual_user_file_service,
        template_service=TemplateService(),
        document_agent_service=test_document_agent_service,
        sql_agent_service=test_sql_agent_service,
        logging_service=LoggingService(),
        documents=[
            Document(
                file_uuid=actual_csv_tsla_historical_user_file_stored_on_user_file_service.file_uuid,
                content=test_csv_tsla_historical_data,
                filename=actual_csv_tsla_historical_user_file_stored_on_user_file_service.filename,
                extension=actual_csv_tsla_historical_user_file_stored_on_user_file_service.extension,
                source_info=SourceInfo(
                    type="widget",
                    name=actual_csv_tsla_historical_user_file_stored_on_user_file_service.filename,
                    uuid=UUID(mock_uuids.ID1.value),
                ),
            )
        ],
        api_keys=None,
    )
    assert len(document_service.downloaded_user_files) == 1

    stored_file = actual_csv_tsla_historical_user_file_stored_on_user_file_service
    document = document_service.documents[0]

    assert document.filename == stored_file.filename
    assert document.extension == stored_file.extension
    assert document.file_uuid == stored_file.file_uuid
    assert (
        document_service.downloaded_user_files[0].content
        == test_csv_tsla_historical_data
    )
    # We don't expect an unstructured file to be loaded for a CSV file
    assert len(document_service.loaded_unstructured_files) == 0


@pytest.mark.asyncio
async def test_document_service_load_csv(
    test_document_service: DocumentService,
    test_csv_tsla_historical_downloaded_user_file: Document,
):
    await test_document_service.load_spreadsheet_like(
        downloaded_user_file=test_csv_tsla_historical_downloaded_user_file
    )

    assert len(test_document_service._sql_table_to_filename_map) == 1
    assert len(test_document_service._sql_agent_service.get_sql_tables_info()) == 1
    assert (
        "tsla"
        in test_document_service._sql_agent_service.get_sql_tables_info()[0].table_name
    )


@pytest.mark.asyncio
async def test_document_service_load_csv_with_integer_column_names(
    test_document_service: DocumentService,
    test_csv_tsla_historical_downloaded_user_file: Document,
):
    test_csv_file = Document(
        content=b"1,2,3\r\na,x,99\r\nb,y,100\r\nc,z,101",
        file_uuid=uuid4(),
        filename="test_csv_file.csv",
        extension="csv",
        source_info=SourceInfo(
            type="widget",
            name="test_csv_file.csv",
            uuid=uuid4(),
        ),
    )
    await test_document_service.load_spreadsheet_like(
        downloaded_user_file=test_csv_file
    )

    # TODO: Should find a way to test / expose this without accessing the
    # private variable / sql agent service
    assert len(test_document_service._sql_table_to_filename_map) == 1
    assert len(test_document_service._sql_agent_service.get_sql_tables_info()) == 1
    assert (
        "test_csv_file"
        in test_document_service._sql_agent_service.get_sql_tables_info()[0].table_name
    )
    assert all(
        column_name
        in test_document_service._sql_agent_service.get_sql_tables_info()[0].sql_schema
        for column_name in ["1", "2", "3"]
    )


@pytest.mark.asyncio
async def test_document_service_load_csv_with_missing_column_names(
    test_document_service: DocumentService,
    test_csv_tsla_historical_downloaded_user_file: Document,
):
    test_csv_file = Document(
        content=b"1,,3\r\na,x,99\r\nb,y,100\r\nc,z,101",
        file_uuid=uuid4(),
        filename="test_csv_file.csv",
        extension="csv",
        source_info=SourceInfo(
            type="widget",
            name="test_csv_file.csv",
            uuid=uuid4(),
        ),
    )
    await test_document_service.load_spreadsheet_like(
        downloaded_user_file=test_csv_file
    )

    # TODO: Should find a way to test / expose this without accessing the
    # private variable / sql agent service
    assert len(test_document_service._sql_table_to_filename_map) == 1
    assert len(test_document_service._sql_agent_service.get_sql_tables_info()) == 1
    assert (
        "test_csv_file"
        in test_document_service._sql_agent_service.get_sql_tables_info()[0].table_name
    )
    assert all(
        column_name
        in test_document_service._sql_agent_service.get_sql_tables_info()[0].sql_schema
        for column_name in ["1", "unnamed", "3"]
    )


@pytest.mark.asyncio
async def test_document_service_load_csv_with_duplicate_column_names(
    test_document_service: DocumentService,
    test_csv_tsla_historical_downloaded_user_file: Document,
):
    test_csv_file = Document(
        content=b"my_column,my_column,my_column\r\na,x,99\r\nb,y,100\r\nc,z,101",
        file_uuid=uuid4(),
        filename="test_csv_file.csv",
        extension="csv",
        source_info=SourceInfo(
            type="widget",
            name="test_csv_file.csv",
            uuid=uuid4(),
        ),
    )
    await test_document_service.load_spreadsheet_like(
        downloaded_user_file=test_csv_file
    )

    assert len(test_document_service._sql_table_to_filename_map) == 1
    assert len(test_document_service._sql_agent_service.get_sql_tables_info()) == 1
    assert (
        "test_csv_file"
        in test_document_service._sql_agent_service.get_sql_tables_info()[0].table_name
    )


@pytest.mark.asyncio
async def test_document_service_load_xlsx(
    test_document_service: DocumentService,
    test_xlsx_management_comp_downloaded_user_file: Document,
):
    await test_document_service.load_spreadsheet_like(
        downloaded_user_file=test_xlsx_management_comp_downloaded_user_file
    )

    # We expect two tables, because the XLSX contains two sheets
    assert len(test_document_service._sql_table_to_filename_map) == 2
    assert len(test_document_service._sql_agent_service.get_sql_tables_info()) == 2
    assert all(
        "management" in table_info.table_name
        for table_info in test_document_service._sql_agent_service.get_sql_tables_info()
    )


@pytest.mark.skip(reason="Integration tests are skipped and will be rethought.")
@pytest.mark.asyncio
@pytest.mark.integration
async def test_document_service_query_unstructured_files_existing_vectordb(
    actual_user_file_service: UserFileService,
    test_pdf_openbb_story_data: bytes,
    actual_pdf_openbb_story_user_file_stored_on_user_file_service: UserFile,
    actual_pdf_openbb_story_associated_vector_db_stored_on_user_file_service: UserFile,
    test_document_agent_service: DocumentAgentService,
    test_template_service: TemplateService,
    mock_uuids: type[MockUUIDs],
):
    document_service = await get_document_service(
        user_file_service=actual_user_file_service,
        template_service=test_template_service,
        document_agent_service=test_document_agent_service,
        sql_agent_service=Mock(),
        logging_service=LoggingService(),
        documents=[
            Document(
                content=test_pdf_openbb_story_data,
                filename=actual_pdf_openbb_story_user_file_stored_on_user_file_service.filename,
                extension=actual_pdf_openbb_story_user_file_stored_on_user_file_service.extension,
                source_info=SourceInfo(
                    type="widget",
                    name=actual_pdf_openbb_story_user_file_stored_on_user_file_service.filename,
                    uuid=UUID(mock_uuids.ID1.value),
                ),
            )
        ],
        api_keys=None,
    )

    results = []
    async for result in document_service.query_unstructured_files(
        query="Which company is fully remote?"
    ):
        results.append(result)
        if isinstance(result, DocumentQueryResult):
            assert "OpenBB" in result.answer
            assert result.citations[0].source_info.type == "widget"
            assert result.citations[0].source_info.name == "openbb_story.pdf"
            assert (
                result.citations[0].details[0]["Page"] == 1
                or result.citations[0].details[0]["Page"] == 5
            )  # noqa: E501
            assert result.citations[0].details[0]["Filename"] == "openbb_story.pdf"

    assert any(isinstance(result, DocumentQueryResult) for result in results)


@pytest.mark.skip(reason="Integration tests are skipped and will be rethought.")
@pytest.mark.asyncio
@pytest.mark.integration
async def test_document_service_query_unstructured_files_need_to_create_vectordb(
    actual_user_file_service: UserFileService,
    test_pdf_openbb_story_data: bytes,
    actual_pdf_openbb_story_user_file_stored_on_user_file_service: UserFile,
    test_template_service: TemplateService,
    test_document_agent_service: DocumentAgentService,
    mock_uuids: type[MockUUIDs],
):
    test_document = Document(
        file_uuid=actual_pdf_openbb_story_user_file_stored_on_user_file_service.file_uuid,
        # We add some random bytes to the end of the file to ensure that the
        # hash changes, forcing a new vector database to be created and uploaded
        # to hub.
        content=test_pdf_openbb_story_data + str(uuid4()).encode(),
        filename=actual_pdf_openbb_story_user_file_stored_on_user_file_service.filename,
        extension=actual_pdf_openbb_story_user_file_stored_on_user_file_service.extension,
        source_info=SourceInfo(
            type="widget",
            name=actual_pdf_openbb_story_user_file_stored_on_user_file_service.filename,
            uuid=UUID(mock_uuids.ID1.value),
        ),
    )

    document_service = await get_document_service(
        user_file_service=actual_user_file_service,
        template_service=test_template_service,
        document_agent_service=test_document_agent_service,
        sql_agent_service=Mock(),
        logging_service=LoggingService(),
        api_keys=None,
        documents=[test_document],
    )

    results = []
    async for result in document_service.query_unstructured_files(
        query="What was the original name of OpenBB?"
    ):
        results.append(result)
        if isinstance(result, DocumentQueryResult):
            assert "gamestonk" in result.answer.lower()
            assert result.citations[0].source_info.name == "openbb_story.pdf"
            assert result.citations[0].source_info.type == "widget"
            assert result.citations[0].details[0]["Page"] in [1, 4]
            assert result.citations[0].details[0]["Filename"] == "openbb_story.pdf"

    assert any(isinstance(result, DocumentQueryResult) for result in results)


@pytest.mark.skip(reason="Integration tests are skipped and will be rethought.")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_document_service_query_structured_files_csv(
    actual_user_file_service: UserFileService,
    test_csv_tsla_historical_downloaded_user_file: Document,
    actual_csv_tsla_historical_user_file_stored_on_user_file_service: UserFile,
    test_sql_agent_service: SqlAgentService,
    test_template_service: TemplateService,
    mock_uuids: type[MockUUIDs],
):
    document_service = await get_document_service(
        # TODO: Can probably rely on the testing user file service
        user_file_service=actual_user_file_service,
        template_service=test_template_service,
        document_agent_service=Mock(),
        sql_agent_service=test_sql_agent_service,
        logging_service=LoggingService(),
        api_keys=None,
        documents=[
            Document(
                content=test_csv_tsla_historical_downloaded_user_file.content,
                file_uuid=actual_csv_tsla_historical_user_file_stored_on_user_file_service.file_uuid,
                filename=actual_csv_tsla_historical_user_file_stored_on_user_file_service.filename,
                extension=actual_csv_tsla_historical_user_file_stored_on_user_file_service.extension,
                source_info=SourceInfo(
                    type="widget",
                    name=actual_csv_tsla_historical_user_file_stored_on_user_file_service.filename,
                    uuid=UUID(mock_uuids.ID1.value),
                    metadata={"filename": "tsla_historical.csv"},
                ),
            )
        ],
    )

    results = []
    async for result in document_service.query_structured_files(
        query="What was the average closing price for TSLA?"
    ):
        results.append(result)
        if isinstance(result, DocumentQueryResult):
            assert "223.78" in result.answer
            assert result.citations[0].source_info.name == "tsla_historical.csv"
            assert result.citations[0].source_info.type == "widget"
            assert result.citations[0].details[0]["Filename"] == "tsla_historical.csv"
        elif isinstance(result, StatusUpdateSSE):
            assert len(result.data.artifacts) == 1

    assert any(isinstance(result, DocumentQueryResult) for result in results)


@pytest.mark.skip(reason="Integration tests are skipped and will be rethought.")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_document_service_query_structured_files_docx(
    actual_user_file_service: UserFileService,
    test_docx_tesla_wikipedia_downloaded_user_file: Document,
    actual_docx_tesla_wikipedia_user_file_stored_on_user_file_service: UserFile,
    test_document_agent_service: DocumentAgentService,
    test_sql_agent_service: SqlAgentService,
    test_template_service: TemplateService,
    mock_uuids: type[MockUUIDs],
):
    document_service = await get_document_service(
        user_file_service=actual_user_file_service,
        template_service=test_template_service,
        document_agent_service=test_document_agent_service,
        sql_agent_service=test_sql_agent_service,
        logging_service=LoggingService(),
        api_keys=None,
        documents=[
            Document(
                content=test_docx_tesla_wikipedia_downloaded_user_file.content,
                file_uuid=actual_docx_tesla_wikipedia_user_file_stored_on_user_file_service.file_uuid,
                filename=actual_docx_tesla_wikipedia_user_file_stored_on_user_file_service.filename,
                extension=actual_docx_tesla_wikipedia_user_file_stored_on_user_file_service.extension,
                source_info=SourceInfo(
                    type="widget",
                    name=actual_docx_tesla_wikipedia_user_file_stored_on_user_file_service.filename,
                    uuid=UUID(mock_uuids.ID1.value),
                    metadata={"filename": "tesla_wikipedia.docx"},
                ),
            )
        ],
    )

    results = []
    async for result in document_service.query_unstructured_files(
        query="Who incorporated Tesla?"
    ):
        results.append(result)
        if isinstance(result, DocumentQueryResult):
            assert "Martin Eberhard" in result.answer
            assert "Marc Tarpenning" in result.answer
            assert result.citations[0].source_info.name == "tesla_wikipedia.docx"
            assert result.citations[0].source_info.type == "widget"
            assert result.citations[0].details[0]["Filename"] == "tesla_wikipedia.docx"

    assert any(isinstance(result, DocumentQueryResult) for result in results)


@pytest.mark.skip(reason="Integration tests are skipped and will be rethought.")
@pytest.mark.asyncio
@pytest.mark.integration
async def test_document_service_query_unstructured_files_multiple_files_mix(
    actual_user_file_service: UserFileService,
    test_txt_hitchhikers_guide_data: bytes,
    test_pdf_openbb_story_data: bytes,
    actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service: UserFile,
    actual_txt_hitchhikers_guide_associated_vector_db_stored_on_user_file_service: UserFile,  # noqa: E501
    actual_pdf_openbb_story_user_file_stored_on_user_file_service: UserFile,
    test_document_agent_service: DocumentAgentService,
    test_template_service: TemplateService,
    mock_uuids: type[MockUUIDs],
):
    document_service = await get_document_service(
        user_file_service=actual_user_file_service,
        template_service=test_template_service,
        document_agent_service=test_document_agent_service,
        sql_agent_service=Mock(),
        logging_service=LoggingService(),
        api_keys=None,
        documents=[
            Document(
                content=test_txt_hitchhikers_guide_data,
                file_uuid=actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.file_uuid,
                filename=actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.filename,
                extension=actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.extension,
                source_info=SourceInfo(
                    type="widget",
                    name=actual_txt_hitchhikers_guide_user_file_stored_on_user_file_service.filename,
                    uuid=UUID(mock_uuids.ID1.value),
                ),
            ),
            Document(
                content=test_pdf_openbb_story_data,
                file_uuid=actual_pdf_openbb_story_user_file_stored_on_user_file_service.file_uuid,
                filename=actual_pdf_openbb_story_user_file_stored_on_user_file_service.filename,
                extension=actual_pdf_openbb_story_user_file_stored_on_user_file_service.extension,
                source_info=SourceInfo(
                    type="widget",
                    name=actual_pdf_openbb_story_user_file_stored_on_user_file_service.filename,
                    uuid=UUID(mock_uuids.ID2.value),
                ),
            ),
        ],
    )

    results = []
    async for result in document_service.query_unstructured_files(
        query="What is the answer to the ultimate question of life?"
    ):
        results.append(result)
        if isinstance(result, DocumentQueryResult):
            assert "42" in result.answer
            assert result.citations[0].source_info.name == "hitchhikers_guide.txt"
            assert result.citations[0].source_info.type == "widget"
            assert result.citations[0].details[0]["Filename"] == "hitchhikers_guide.txt"

    assert any(isinstance(result, DocumentQueryResult) for result in results)

    results = []
    async for result in document_service.query_unstructured_files(
        query="What was the original name of OpenBB?"
    ):
        results.append(result)
        if isinstance(result, DocumentQueryResult):
            assert "gamestonk" in result.answer.lower()
            assert result.citations[0].source_info.name == "openbb_story.pdf"
            assert result.citations[0].source_info.type == "widget"
            assert result.citations[0].details[0]["Filename"] == "openbb_story.pdf"
            assert result.citations[0].details[0]["Page"] == 1

    assert any(isinstance(result, DocumentQueryResult) for result in results)


@pytest.mark.skip(reason="Integration tests are skipped and will be rethought.")
@pytest.mark.asyncio
@pytest.mark.integration
async def test_document_service_query_image_file(
    actual_user_file_service: UserFileService,
    test_png_table_image_data: bytes,
    actual_png_table_image_user_file_stored_on_user_file_service: UserFile,
    test_document_agent_service: DocumentAgentService,
    test_template_service: TemplateService,
    mock_uuids: type[MockUUIDs],
):
    document_service = await get_document_service(
        user_file_service=actual_user_file_service,
        template_service=test_template_service,
        document_agent_service=test_document_agent_service,
        sql_agent_service=Mock(),
        logging_service=LoggingService(),
        api_keys=None,
        documents=[
            Document(
                content=test_png_table_image_data,
                file_uuid=actual_png_table_image_user_file_stored_on_user_file_service.file_uuid,
                filename=actual_png_table_image_user_file_stored_on_user_file_service.filename,
                extension=actual_png_table_image_user_file_stored_on_user_file_service.extension,
                source_info=SourceInfo(
                    type="widget",
                    name=actual_png_table_image_user_file_stored_on_user_file_service.filename,
                    uuid=UUID(mock_uuids.ID1.value),
                ),
            ),
        ],
    )

    results = []
    async for result in document_service.query_unstructured_files(
        query="What was the total revenues for the first quarter of 2024?"
    ):
        results.append(result)
        if isinstance(result, DocumentQueryResult):
            assert "21,301" in result.answer
            assert result.citations[0].source_info.name == "table.png"
            assert result.citations[0].source_info.type == "widget"
            assert result.citations[0].details[0]["Filename"] == "table.png"

    assert any(isinstance(result, DocumentQueryResult) for result in results)
