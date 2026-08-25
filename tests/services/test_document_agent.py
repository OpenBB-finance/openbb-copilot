from unittest.mock import Mock

import pytest
from openbb_ai.models import SourceInfo

from openbb_ada.models import (
    Document,
    DocumentAgentQueryResult,
)
from openbb_ada.services import DocumentAgentService, TemplateService
from openbb_ada.vector_db import VectorDb


def _build_test_html_document() -> Document:
    html_content = """
    <!doctype html>
    <html>
      <head>
        <title>Tesla Overview</title>
        <style>body { color: red; }</style>
        <script>console.log("ignore me")</script>
      </head>
      <body>
        <h1>Tesla Overview</h1>
        <p>Tesla, Inc. was incorporated by <strong>Martin Eberhard</strong> and
        <em>Marc Tarpenning</em> in July 2003.</p>
        <ul>
          <li>Founded in 2003.</li>
          <li>Headquartered in Austin.</li>
        </ul>
      </body>
    </html>
    """.encode()
    return Document(
        content=html_content,
        filename="tesla_overview.html",
        extension="html",
        source_info=SourceInfo(
            uuid="00000000-0000-0000-0000-000000000123",
            origin="test_origin",
            widget_id="tesla_overview",
            type="widget",
            name="Tesla Overview",
            description="Tesla HTML test document.",
            metadata={"filename": "tesla_overview.html"},
        ),
    )


@pytest.mark.asyncio
async def test_document_agent_service_init(
    test_template_service: TemplateService,
):
    document_service = DocumentAgentService(
        template_service=test_template_service,
        logging_service=Mock(),
        openai_api_key="test_key",
    )
    assert document_service is not None
    assert document_service._openai_api_key == "test_key"


@pytest.mark.asyncio
async def test_document_agent_service_load_pdf(
    test_document_agent_service: DocumentAgentService,
    test_pdf_openbb_story_downloaded_user_file: Document,
):
    actual_result: VectorDb = (
        await test_document_agent_service.load_unstructured_document(
            document=test_pdf_openbb_story_downloaded_user_file
        )
    )
    assert isinstance(actual_result, VectorDb)
    assert (
        actual_result.docs[0].metadata["name"]
        == test_pdf_openbb_story_downloaded_user_file.filename
    )
    assert len(test_document_agent_service._filename_to_vector_db_map) == 1
    assert (
        test_pdf_openbb_story_downloaded_user_file.filename
        in test_document_agent_service._filename_to_vector_db_map
    )
    assert (
        test_pdf_openbb_story_downloaded_user_file
        in test_document_agent_service.loaded_files
    )


@pytest.mark.asyncio
async def test_document_agent_service_load_html_strips_markup(
    test_document_agent_service: DocumentAgentService,
    monkeypatch: pytest.MonkeyPatch,
):
    html_document = _build_test_html_document()

    async def fake_create_vector_db(documents):
        vector_db = VectorDb()
        vector_db._docs = documents
        return vector_db

    monkeypatch.setattr(
        test_document_agent_service,
        "create_vector_db",
        fake_create_vector_db,
    )

    actual_result = await test_document_agent_service.load_unstructured_document(
        document=html_document
    )

    assert isinstance(actual_result, VectorDb)
    stored_text = "\n".join(doc.page_content for doc in actual_result.docs)
    assert "Tesla Overview" in stored_text
    assert "Martin Eberhard" in stored_text
    assert "Marc Tarpenning" in stored_text
    assert "Founded in 2003." in stored_text
    assert "<html" not in stored_text
    assert "console.log" not in stored_text
    assert "body { color: red; }" not in stored_text


@pytest.mark.asyncio
async def test_document_agent_service_peek_html_returns_plaintext(
    test_document_agent_service: DocumentAgentService,
):
    html_document = _build_test_html_document()
    await test_document_agent_service.load_unstructured_document(
        document=html_document,
        vector_db=VectorDb(),
    )

    result = await test_document_agent_service._llm_peek_file(html_document.filename)

    assert "Tesla Overview" in result
    assert "Martin Eberhard" in result
    assert "Marc Tarpenning" in result
    assert "<strong>" not in result
    assert "console.log" not in result
    assert test_document_agent_service._retrieved_document_chunks[-1].metadata[
        "name"
    ] == (html_document.filename)


@pytest.mark.asyncio
async def test_document_agent_service_summarize_html_uses_plaintext(
    test_document_agent_service: DocumentAgentService,
    monkeypatch: pytest.MonkeyPatch,
):
    html_document = _build_test_html_document()
    await test_document_agent_service.load_unstructured_document(
        document=html_document,
        vector_db=VectorDb(),
    )
    captured: dict[str, str] = {}

    def fake_chatprompt(*args, **kwargs):
        del args, kwargs

        def decorator(func):
            del func

            async def fake_summarize_text(text: str) -> str:
                captured["text"] = text
                return "Tesla was incorporated by Martin Eberhard and Marc Tarpenning."

            return fake_summarize_text

        return decorator

    monkeypatch.setattr(
        "openbb_ada.services.document_agent.chatprompt", fake_chatprompt
    )

    summary = await test_document_agent_service._summarize_document(
        "Summarize the founders.",
        html_document.filename,
    )

    assert "Created summary artifact:" in summary
    assert "Martin Eberhard" in summary
    assert "Marc Tarpenning" in summary
    assert "Tesla Overview" in captured["text"]
    assert "<html" not in captured["text"]
    assert "console.log" not in captured["text"]
    assert "body { color: red; }" not in captured["text"]


@pytest.mark.asyncio
async def test_document_agent_service_query_pdf(
    test_document_agent_service: DocumentAgentService,
    test_pdf_openbb_story_downloaded_user_file: Document,
):
    await test_document_agent_service.load_unstructured_document(
        document=test_pdf_openbb_story_downloaded_user_file
    )
    actual_result: DocumentAgentQueryResult = await test_document_agent_service.query(
        query="Who is the author of the file, and what was the name of the company he started?"  # noqa: E501
    )

    assert isinstance(actual_result, DocumentAgentQueryResult)
    assert "company" in actual_result.answer
    assert "OpenBB" in actual_result.answer


@pytest.mark.asyncio
async def test_document_agent_service_query_search(
    test_document_agent_service: DocumentAgentService,
    test_pdf_tsla_10q_downloaded_user_file: Document,
):
    await test_document_agent_service.load_unstructured_document(
        document=test_pdf_tsla_10q_downloaded_user_file
    )

    actual_result: DocumentAgentQueryResult = await test_document_agent_service.query(
        query="What was the total revenue for TSLA in Q1 2024?"
    )
    assert isinstance(actual_result, DocumentAgentQueryResult)
    assert "$21.30" in actual_result.answer or "$21,30" in actual_result.answer
    assert len(actual_result.citations) >= 1
    assert actual_result.citations[0].source_info.name == "tsla_10q.pdf"
    assert actual_result.citations[0].source_info.type == "widget"
    assert actual_result.citations[0].details[0]["Filename"] == "tsla_10q.pdf"
    assert actual_result.citations[0].details[0]["Page"] in [25, 27]


@pytest.mark.asyncio
async def test_document_agent_service_query_images(
    test_document_agent_service: DocumentAgentService,
    test_jpg_lion_image_downloaded_user_file: Document,
    test_jpg_whale_image_downloaded_user_file: Document,
):
    await test_document_agent_service.load_image(
        document=test_jpg_lion_image_downloaded_user_file
    )
    await test_document_agent_service.load_image(
        document=test_jpg_whale_image_downloaded_user_file
    )

    actual_result: DocumentAgentQueryResult = await test_document_agent_service.query(
        query="Which animals are in the images? And what is their colour?"
    )
    assert "lion" in actual_result.answer.lower()
    assert "whale" in actual_result.answer.lower()
    assert len(actual_result.citations) >= 2
    assert any(
        "animal_1.jpg" in citation.source_info.name  # type: ignore
        for citation in actual_result.citations
    )
    assert any(
        "animal_2.jpg" in citation.source_info.name  # type: ignore
        for citation in actual_result.citations
    )
    assert any(
        "animal_1.jpg" == citation.details[0]["Filename"]  # type: ignore
        for citation in actual_result.citations
    )
    assert any(
        "animal_2.jpg" == citation.details[0]["Filename"]  # type: ignore
        for citation in actual_result.citations
    )


@pytest.mark.asyncio
async def test_document_agent_service_query_multiple_files(
    test_document_agent_service: DocumentAgentService,
    test_pdf_openbb_story_downloaded_user_file: Document,
    test_pdf_tsla_10q_downloaded_user_file: Document,
):
    await test_document_agent_service.load_unstructured_document(
        document=test_pdf_openbb_story_downloaded_user_file
    )
    await test_document_agent_service.load_unstructured_document(
        document=test_pdf_tsla_10q_downloaded_user_file
    )

    actual_result: DocumentAgentQueryResult = await test_document_agent_service.query(
        "Who wrote the blog article? And what was the total gross profit of TSLA in Q1 2024? You must search both documents."  # noqa: E501
    )

    assert "Didier Lopes" in actual_result.answer
    assert (
        "$3,696" in actual_result.answer.replace(".", ",")
        or "$3.7 b" in actual_result.answer
    )
    assert len(actual_result.citations) >= 2

    # Check that both expected citations are present
    # (order can vary due to LLM non-determinism)
    citation_names = [citation.source_info.name for citation in actual_result.citations]
    assert "openbb_story.pdf" in citation_names
    assert "tsla_10q.pdf" in citation_names

    # Find the citations by filename to check their specific properties
    openbb_citation = next(
        c for c in actual_result.citations if c.source_info.name == "openbb_story.pdf"
    )
    tsla_citation = next(
        c for c in actual_result.citations if c.source_info.name == "tsla_10q.pdf"
    )

    assert openbb_citation.details[0]["Page"] in [1]
    assert tsla_citation.details[0]["Page"] in [28]
    assert openbb_citation.details[0]["Filename"] == "openbb_story.pdf"
    assert tsla_citation.details[0]["Filename"] == "tsla_10q.pdf"


@pytest.mark.asyncio
async def test_document_agent_service_query_summarize(
    test_document_agent_service: DocumentAgentService,
    test_pdf_openbb_story_downloaded_user_file: Document,
):
    await test_document_agent_service.load_unstructured_document(
        document=test_pdf_openbb_story_downloaded_user_file
    )
    actual_result: DocumentAgentQueryResult = await test_document_agent_service.query(
        query="Summarize the attached file, focusing on all people and investors involved."  # noqa: E501
    )
    # Text summaries should not create artifacts - content should be in the answer
    assert isinstance(actual_result.artifacts, list)
    assert len(actual_result.artifacts) == 0
    assert "Didier" in actual_result.answer and "Lopes" in actual_result.answer
    assert "OpenBB" in actual_result.answer
    assert "OSS Capital" in actual_result.answer
    assert actual_result.citations[0].source_info.name == "openbb_story.pdf"
    assert actual_result.citations[0].source_info.type == "widget"
    assert actual_result.citations[0].details[0]["Filename"] == "openbb_story.pdf"


@pytest.mark.asyncio
async def test_document_agent_service_query_multiple_files_summarize_concurrently(
    test_document_agent_service: DocumentAgentService,
    test_pdf_openbb_story_downloaded_user_file: Document,
    test_pdf_tsla_10q_downloaded_user_file: Document,
):
    await test_document_agent_service.load_unstructured_document(
        document=test_pdf_openbb_story_downloaded_user_file
    )
    await test_document_agent_service.load_unstructured_document(
        document=test_pdf_tsla_10q_downloaded_user_file
    )

    summary_goal = "Summarize the content of the documents"

    filenames = [
        test_pdf_openbb_story_downloaded_user_file.filename,
        test_pdf_tsla_10q_downloaded_user_file.filename,
    ]
    summaries = await test_document_agent_service._llm_summarize_documents(
        summary_goal, filenames
    )

    assert len(summaries) == 2

    artifact_id_1 = summaries[0].split("`")[1]
    artifact_id_2 = summaries[1].split("`")[1]

    assert artifact_id_1 in test_document_agent_service._artifacts
    assert artifact_id_2 in test_document_agent_service._artifacts

    assert len(test_document_agent_service._artifacts[artifact_id_1].content) > 100
    assert len(test_document_agent_service._artifacts[artifact_id_2].content) > 100

    assert (
        f"summary_{test_pdf_openbb_story_downloaded_user_file.filename.replace('.', '_')}".lower()[  # noqa: E501
            :40
        ]
        in artifact_id_1
    )
    assert (
        f"summary_{test_pdf_tsla_10q_downloaded_user_file.filename.replace('.', '_')}".lower()[  # noqa: E501
            :40
        ]
        in artifact_id_2
    )


@pytest.mark.asyncio
async def test_document_agent_no_citations_when_content_not_found(
    test_document_agent_service: DocumentAgentService,
    test_pdf_openbb_story_downloaded_user_file: Document,
):
    """Test that "not found" responses do not include unrelated citations.

    This test verifies that when the Document Agent cannot find the requested
    information in a document, it should return an answer indicating the
    information was not found. Citations are optional across model families:
    some models return none, while others cite a summary artifact confirming
    the information is absent. If citations are present, they must still point
    to the queried document.
    """
    await test_document_agent_service.load_unstructured_document(
        document=test_pdf_openbb_story_downloaded_user_file
    )

    # Query for something that definitely doesn't exist in the OpenBB story PDF
    actual_result: DocumentAgentQueryResult = await test_document_agent_service.query(
        query="What was the exact price of Bitcoin on March 15, 2019 at 3:45 PM UTC?"
    )

    assert isinstance(actual_result, DocumentAgentQueryResult)
    # The answer should indicate the information was not found
    assert any(
        phrase in actual_result.answer.lower()
        for phrase in [
            "not found",
            "couldn't find",
            "could not find",
            "no information",
            "not mentioned",
            "does not contain",
            "doesn't contain",
            "not in the document",
            "not available",
            "unable to find",
        ]
    )
    # Citations are optional for "not found" answers, but when present they
    # must still reference the queried file (no unrelated sources).
    for citation in actual_result.citations:
        assert citation.source_info.type == "widget"
        assert citation.details
        assert any(
            detail.get("Filename")
            == test_pdf_openbb_story_downloaded_user_file.filename
            for detail in citation.details
        )
