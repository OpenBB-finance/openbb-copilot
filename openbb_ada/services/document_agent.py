import asyncio
import io
import re
import uuid
from copy import deepcopy
from typing import (
    Annotated,
    Any,
    Callable,
    Coroutine,
    List,
)

import httpx
import pdfplumber
import tiktoken
from docx import Document as DocxDocument
from magentic import (
    AssistantMessage,
    FunctionCall,
    FunctionResultMessage,
    SystemMessage,
    UserMessage,
    chatprompt,
)
from magentic.vision import UserImageMessage
from openbb_ai.models import (
    CitationHighlightBoundingBox,
)
from pydantic import AliasChoices, Field

from .. import constants
from ..errors import (
    FunctionCallError,
)
from ..models import (
    Citation,
    CopilotArtifact,
    Document,
    DocumentAgentArtifact,
    DocumentAgentQueryResult,
    DocumentSourceInfo,
    RawObjectDataFormat,
    RetrievedDocumentChunk,
    SourceInfo,
    Word,
    WordGroup,
)
from ..pdf import Pdf
from ..utils.ai import get_llm
from ..utils.html_text import extract_text_from_html
from ..utils.utils import (
    extract_text_chunks_from_docx,
    find_longest_common_substring_indices,
    retry_on_exception,
    sanitize_str,
    split_text,
)
from ..vector_db import VectorDb, VectorDbDocument
from ._logging import LoggingService, logfire
from .template import TemplateService


class DocumentAgentService:
    IMAGE_FILE_EXTENSIONS = constants.IMAGE_FILE_EXTENSIONS
    UNSTRUCTURED_FILE_EXTENSIONS = constants.UNSTRUCTURED_FILE_EXTENSIONS

    def __init__(
        self,
        template_service: TemplateService,
        logging_service: LoggingService,
        openai_api_key: str | None,
    ):
        self._template_service = template_service
        self._logging_service = logging_service
        self._openai_api_key = openai_api_key
        self._loaded_files: list[Document] = []
        self._filename_to_vector_db_map: dict[str, VectorDb] = {}
        self._retrieved_document_chunks: list[RetrievedDocumentChunk] = []
        self._artifacts: dict[str, DocumentAgentArtifact] = {}

    @property
    def loaded_files(self) -> list[Document]:
        return self._loaded_files

    @staticmethod
    async def create_vector_db(documents: list[VectorDbDocument]) -> VectorDb:
        vector_db = VectorDb()
        await vector_db.add(documents=documents)
        return vector_db

    def _get_model(self, **kwargs):
        return get_llm(api_key=self._openai_api_key, **kwargs)

    async def _load_txt(self, document: Document) -> VectorDb:
        text = document.content.decode()
        if document.extension == "html":
            text = extract_text_from_html(text)
        split_text_list = split_text(text)
        self._logging_service.debug(
            "Document Agent: Preparing %d text chunks for %s",
            len(split_text_list),
            document.filename,
        )
        doc_chunks = []
        for i, _text in enumerate(split_text_list):
            metadata = {
                "document_hash": document.hash,
                "name": document.filename,
                "chunk": i,
            }
            doc = VectorDbDocument(page_content=_text, metadata=metadata)  # type: ignore
            doc_chunks.append(doc)
        try:
            vector_db = await self.create_vector_db(documents=doc_chunks)
        except ValueError as err:
            self._logging_service.error(
                "Unable to create vector database for %s: ",
                document.filename,
                err,
            )
            vector_db = VectorDb()
        return vector_db

    async def _load_pdf(self, document: Document) -> VectorDb:
        pdf = Pdf(content=document.content)
        doc_chunks = []
        for i, page_content in enumerate(pdf.get_text()):
            if len(page_content) == 0:
                continue
            page_content = page_content.replace("\t", " ")
            metadata = {
                "document_hash": document.hash,
                "name": document.filename,
                "page_number": i + 1,
            }
            chunk = VectorDbDocument(page_content=page_content, metadata=metadata)
            doc_chunks.append(chunk)
        self._logging_service.debug(
            "Document Agent: Preparing %d PDF chunks for %s",
            len(doc_chunks),
            document.filename,
        )
        try:
            vector_db = await self.create_vector_db(documents=doc_chunks)
        except ValueError as err:
            self._logging_service.error(
                "Unable to create vector database for %s: ",
                document.filename,
                err,
            )
            vector_db = VectorDb()
        return vector_db

    async def _load_docx(self, document: Document) -> VectorDb:
        doc = DocxDocument(io.BytesIO(document.content))
        text_chunks = extract_text_chunks_from_docx(doc)
        self._logging_service.debug(
            "Document Agent: Preparing %d DOCX chunks for %s",
            len(text_chunks),
            document.filename,
        )
        doc_chunks = []
        for i, text_chunk in enumerate(text_chunks):
            # There is no way to get the paragraphs, tables and pages in order using docx.  # noqa: E501
            # This creates a limitation for sentence level highlights.
            # Currently, the workspace also does not support pages for .docx files.
            # A possible workaround that solved the pages problem is to use LibreOffice
            # to convert .docx to .pdf and then use our PDF pipeline to process it.
            metadata = {
                "document_hash": document.hash,
                "name": document.filename,
                "chunk": i,
            }
            chunk = VectorDbDocument(page_content=text_chunk, metadata=metadata)
            doc_chunks.append(chunk)
        try:
            vector_db = await self.create_vector_db(documents=doc_chunks)
        except ValueError as err:
            self._logging_service.error(
                "Unable to create vector database for %s: ",
                document.filename,
                err,
            )
            vector_db = VectorDb()
        return vector_db

    async def load_unstructured_document(
        self,
        document: Document,
        vector_db: VectorDb | None = None,
    ) -> VectorDb:
        self._logging_service.debug(
            "Document Agent: Loading %s document %s (cached_vector_db=%s)",
            document.extension,
            document.filename,
            vector_db is not None,
        )
        if vector_db:
            self._loaded_files.append(document)
            self._filename_to_vector_db_map[document.filename] = vector_db
            return vector_db
        else:
            if document.extension in ["txt", "md", "html"]:
                vector_db = await self._load_txt(document)
            elif document.extension == "pdf":
                vector_db = await self._load_pdf(document)
            elif document.extension == "docx":
                vector_db = await self._load_docx(document)
            if vector_db is None:
                raise ValueError(f"Unable to load document: {document.filename}")
            self._loaded_files.append(document)
            self._filename_to_vector_db_map[document.filename] = vector_db
            return vector_db

    async def load_image(self, document: Document) -> None:
        self._loaded_files.append(document)

    def has_images(self) -> bool:
        return any(
            file_.extension in self.IMAGE_FILE_EXTENSIONS
            for file_ in self._loaded_files
        )

    def has_documents(self) -> bool:
        return any(
            file_.extension in self.UNSTRUCTURED_FILE_EXTENSIONS
            for file_ in self._loaded_files
        )

    @logfire.instrument("DocumentAgentService.query")
    async def query(
        self,
        query: str,
    ) -> DocumentAgentQueryResult:
        self._logging_service.debug("Document Agent: Entering agent loop...")
        self._logging_service.debug("Document Agent: Query: %s", query)
        self._logging_service.debug(
            "Document Agent: Loaded files: %s",
            [file_.filename for file_ in self._loaded_files],
        )
        messages: list[
            SystemMessage | UserMessage | AssistantMessage | FunctionResultMessage
        ] = [
            SystemMessage(
                self._template_service.render_copilot_document_agent_system_prompt(
                    documents=self._loaded_files,
                    artifacts=self._artifacts,
                )
            ),
            UserMessage("{query}"),
        ]

        functions: list[Callable[..., Coroutine[Any, Any, Any]]] = [
            self._llm_complete,
        ]
        if self.has_images():
            functions += [
                self._llm_query_image,
            ]
        if self.has_documents():
            functions += [
                self._llm_peek_file,
                self._llm_search_document,
                self._llm_summarize_documents,
            ]
        self._logging_service.debug(
            "Document Agent: Available tools: %s",
            [function.__name__ for function in functions],
        )

        call_count = 0
        response = None
        result = None
        while call_count < 10:
            call_count += 1

            @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
            @chatprompt(
                *messages,
                functions=functions,
                model=self._get_model(),
            )
            async def _llm(query: str) -> FunctionCall: ...  # type: ignore

            self._logging_service.debug("Document Agent: Begin loop.")
            self._logging_service.debug("Document Agent: Executing LLM call...")
            try:
                response = await _llm(query=query)
            # Sometimes we encounter a tool call to a tool that doesn't exist.
            # Let's give the model a change to try again.
            # TODO: We might consider making this more robust and sending an
            # error message to the model in future.
            except ValueError as err:
                self._logging_service.error("Document Agent: LLM call failed: %s", err)
                continue
            if isinstance(response, FunctionCall):
                try:
                    self._logging_service.debug(
                        "Document Agent: LLM response is a function call: %s", response
                    )
                    if response._function.__name__ == self._llm_complete.__name__:
                        self._logging_service.debug(
                            "Document Agent: `complete` function called. Returning...",
                        )
                        return await response()
                    self._logging_service.debug("Document Agent: Calling function...")
                    result = await response()
                    self._logging_service.debug(
                        "Document Agent: Appending function call and result to messages..."  # noqa: E501
                    )
                except FunctionCallError as err:
                    self._logging_service.error(
                        "Document Agent: Function call failed: %s", err
                    )
                    messages.append(AssistantMessage(response))
                    messages.append(
                        FunctionResultMessage(
                            content=sanitize_str(str(err)), function_call=response
                        )
                    )
                    continue

                messages.append(AssistantMessage(response))
                messages.append(
                    FunctionResultMessage(
                        content=sanitize_str(str(result)), function_call=response
                    )
                )
                self._logging_service.debug("Document Agent: End loop.")
        self._logging_service.warning(
            "Document Agent: LLM failed to complete query after 10 attempts...exiting."  # noqa: E501
        )

        return DocumentAgentQueryResult(
            answer="I was unable to complete the query after reaching my iteration limit.",  # noqa: E501
        )

    def _get_file_by_name(self, filename: str) -> Document | None:
        for file_ in self._loaded_files:
            if file_.filename == filename:
                return file_
        return None

    def _get_vector_db_by_associated_filename(self, filename: str) -> VectorDb | None:
        return self._filename_to_vector_db_map.get(filename)

    def _count_tokens(self, text: str) -> int:
        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))

    def _pdf_contains_text(self, file: Document) -> bool:
        # Not the most efficient solution, but performance shouldn't be too bad.
        # This problem will hopefully go away when we support scanned documents
        pdf = Pdf(content=file.content)
        pdf_text = "".join(pdf.get_text())
        if pdf_text.strip():
            return True
        return False

    def _extract_text_from_document(self, document: Document) -> str:
        if document.extension == "pdf":
            pdf = Pdf(content=document.content)
            return "\n".join(pdf.get_text())
        if document.extension in ["txt", "md"]:
            return document.content.decode()
        if document.extension == "html":
            return extract_text_from_html(document.content.decode())
        if document.extension == "docx":
            doc = DocxDocument(io.BytesIO(document.content))
            text_chunks = extract_text_chunks_from_docx(doc)
            return "\n".join(text_chunks)
        raise FunctionCallError(f"Unsupported file type: {document.extension}")

    async def _llm_peek_image(self, filename: str) -> str:
        downloaded_file = self._get_file_by_name(filename)
        if downloaded_file is None:
            raise FunctionCallError(f"Unable to find file: {filename}")

        @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
        @chatprompt(
            SystemMessage("{query}"),
            UserImageMessage(downloaded_file.content),
            model=self._get_model(
                model=constants.OPENBB_AGENT_MODEL_SMALL,
                max_completion_tokens=2048,
                temperature=0.0,
            ),
        )
        async def _llm(query: str) -> str: ...  # type: ignore[empty-body]

        peek_query = (
            "Very briefly describe the image in a way that is helpful "
            "for a user to understand what kind of content it contains."
        )

        return await _llm(query=peek_query)

    async def _llm_peek_file(self, filename: str) -> str:
        self._logging_service.debug("Document Agent: Peeking file %s", filename)
        downloaded_file = self._get_file_by_name(filename)
        if downloaded_file is None:
            raise FunctionCallError(f"Unable to find file: {filename}")
        if downloaded_file.extension == "pdf":
            if not self._pdf_contains_text(downloaded_file):
                raise FunctionCallError(
                    f"Unable to query {downloaded_file.filename} because it has no text."  # noqa: E501
                )
            pdf = Pdf(content=downloaded_file.content)
            result = "\n".join(pdf.get_text()[:3])
        elif downloaded_file.extension in ["txt", "md", "html", "docx"]:
            result = self._extract_text_from_document(downloaded_file)[:1000]
        elif downloaded_file.extension in ["png", "jpg", "jpeg"]:
            result = await self._llm_peek_image(filename=downloaded_file.filename)
        else:
            result = ""

        # We add the peeked chunk to the list of retrieved chunks so that it can
        # be used in the `complete` function when providing citations.
        retrieved_chunk = RetrievedDocumentChunk(
            page_content=result,
            metadata={
                "document_hash": downloaded_file.hash,
                "name": downloaded_file.filename,
            },
        )
        self._retrieved_document_chunks.append(retrieved_chunk)

        result = self._template_service.render_copilot_document_agent_retrieved_document_chunks(  # noqa: E501
            retrieved_document_chunks=[retrieved_chunk]
        )
        return result

    async def _llm_query_image(self, query: str, filename: str) -> str:
        downloaded_file = self._get_file_by_name(filename)
        if downloaded_file is None:
            raise FunctionCallError(f"Unable to find file: {filename}")

        @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
        @chatprompt(
            SystemMessage("{query}"),
            UserImageMessage(downloaded_file.content),
            model=self._get_model(
                model=constants.OPENBB_AGENT_MODEL_MAIN,
                max_completion_tokens=4096,
                temperature=0.0,
            ),
        )
        async def _llm(query: str) -> str: ...  # type: ignore[empty-body]

        result = await _llm(query=query)

        # We use a "trick" and treat the image to appear as if we retrieved a
        # chunk from it, like we would a document loaded into a vectorDB.
        retrieved_chunk = RetrievedDocumentChunk(
            page_content=result,
            metadata={
                "document_hash": downloaded_file.hash,
                "name": downloaded_file.filename,
            },
        )
        self._retrieved_document_chunks.append(retrieved_chunk)
        return self._template_service.render_copilot_document_agent_retrieved_document_chunks(  # noqa: E501
            retrieved_document_chunks=[retrieved_chunk]
        )

    async def _llm_search_document(
        self,
        query: Annotated[
            str,
            Field(validation_alias=AliasChoices("query", "Query")),
        ],
        filename: Annotated[
            str | None,
            Field(validation_alias=AliasChoices("filename", "Filename")),
        ] = None,
    ) -> str:
        """Search documents for specific information.

        Returns search results with content_id values. Use these content_ids when
        creating DocumentSourceInfo objects in _llm_complete to enable citations.

        Parameters
        ----------
        query : str
            Keywords and phrases to search for.
        filename : str or None
            Optional - search only this specific file. If None, searches all documents.

        Returns
        -------
        str
            Search results containing content_id values and matching text.
        """
        self._logging_service.debug(
            "Document Agent: Searching documents (filename=%s, query_length=%d)",
            filename or "ALL",
            len(query),
        )
        if filename is None:  # Search all documents
            vector_dbs = list(self._filename_to_vector_db_map.values())
            if not vector_dbs:
                raise FunctionCallError("No documents loaded.")
            else:
                vector_db = VectorDb()
                for db in vector_dbs:
                    vector_db.merge_from(deepcopy(db))
        else:
            downloaded_file = self._get_file_by_name(filename)
            if downloaded_file and downloaded_file.extension == "pdf":
                if not self._pdf_contains_text(downloaded_file):
                    raise FunctionCallError(
                        f"Unable to query {downloaded_file.filename} because it has no text."  # noqa: E501
                    )
            vector_db_or_none = self._get_vector_db_by_associated_filename(filename)
            if vector_db_or_none is None:
                raise FunctionCallError(f"Unable to find file: {filename}")
            vector_db = vector_db_or_none

        vector_db_documents = await vector_db.search(query=query)
        self._logging_service.debug(
            "Document Agent: Search returned %d chunks for %s",
            len(vector_db_documents),
            filename or "ALL",
        )
        retrieved_chunks = [
            RetrievedDocumentChunk(**chunk.model_dump())
            for chunk in vector_db_documents
        ]
        self._retrieved_document_chunks.extend(retrieved_chunks)

        results = self._template_service.render_copilot_document_agent_retrieved_document_chunks(  # noqa: E501
            retrieved_document_chunks=retrieved_chunks
        )
        return results

    async def _llm_summarize_documents(
        self,
        # AliasChoices accommodates models that emit PascalCase or space-separated
        # parameter names instead of snake_case when invoking tool calls.
        summary_goal: Annotated[
            str,
            Field(
                validation_alias=AliasChoices(
                    "summary_goal", "Summary Goal", "SummaryGoal"
                )
            ),
        ],
        filenames: Annotated[
            List[str],
            Field(validation_alias=AliasChoices("filenames", "Filenames", "Files")),
        ],
    ) -> List[str]:
        """Summarize documents with a particular goal in mind.

        Use this for general overviews and summaries. Returns artifact IDs that
        you should pass to _llm_complete as DocumentSourceInfo(content_id=artifact_id)
        to create document-level citations.

        Parameters
        ----------
        summary_goal : str
            What you want to learn from the summary
        filenames : List[str]
            Documents to summarize

        Important
        ---------
        When calling this function, use the exact snake_case parameter names
        shown above (``summary_goal`` and ``filenames``).

        Returns
        -------
        List[str]
            Summary results with artifact IDs for citations.
        """
        self._logging_service.debug(
            "Document Agent: Summarizing %d document(s) with goal_length=%d",
            len(filenames),
            len(summary_goal),
        )

        tasks = [
            self._summarize_document(summary_goal, filename) for filename in filenames
        ]
        return await asyncio.gather(*tasks)

    async def _summarize_document(self, summary_goal: str, filename: str) -> str:
        """Summarize a document with a particular goal in mind.

        Creates internal artifact for citations but returns artifact reference.
        The artifact won't be sent to user due to content filtering."""
        downloaded_file = self._get_file_by_name(filename)
        if downloaded_file is None:
            raise FunctionCallError(f"Unable to find file: {filename}")
        if downloaded_file.extension == "pdf":
            if not self._pdf_contains_text(downloaded_file):
                raise FunctionCallError(
                    f"Unable to query {downloaded_file.filename} because it has no text."  # noqa: E501
                )
        text = self._extract_text_from_document(downloaded_file)

        @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
        @chatprompt(
            SystemMessage(
                f"Summarize the following text with the following goal in mind: {summary_goal}. Extract all information relevant to the goal."  # noqa: E501
            ),
            UserMessage("{text}"),
            model=self._get_model(
                model=constants.OPENBB_AGENT_MODEL_SMALL, temperature=0.0
            ),
        )
        async def _summarize_text(text: str) -> str: ...  # type: ignore[empty-body]

        self._logging_service.debug(
            "Document Agent: Summarizing %s with %d characters (goal_length=%d)",
            filename,
            len(text),
            len(summary_goal),
        )

        call_count = 0
        # 4 characters ~= 1 token
        while self._count_tokens(text) > 16_000 * 4 and call_count < 5:
            self._logging_service.debug("Document Agent: Splitting text into chunks...")
            chunks = split_text(text, chunk_size=16_000 * 4, chunk_overlap=100 * 4)
            self._logging_service.debug(
                "Document Agent: Summarizing %d chunks for %s",
                len(chunks),
                filename,
            )
            tasks = [_summarize_text(chunk) for chunk in chunks]
            summaries = await asyncio.gather(*tasks)
            text = "\n".join(summaries)
            self._logging_service.debug(
                "Document Agent: Intermediate summary reduced to %d characters",
                len(text),
            )

        self._logging_service.debug(
            "Document Agent: Creating final summary for %s",
            filename,
        )
        result = await _summarize_text(text)
        self._logging_service.debug(
            "Document Agent: Created final summary of length %d for %s",
            len(result),
            filename,
        )
        # Create artifact for internal reference (needed for citations)
        # but _should_create_artifact() will prevent it from being sent to user
        artifact_id = (
            f"summary_{filename.replace('.', '_').replace(' ', '_')}".lower()[:40]
            + "_"
            + str(uuid.uuid4())[:5]
        )

        self._artifacts[artifact_id] = DocumentAgentArtifact(
            artifact_id=artifact_id,
            document_hash=downloaded_file.hash,
            content=result,
        )

        # Return both artifact reference and content for LLM to use
        # The artifact won't be sent to user due to _should_create_artifact() logic
        return f"Created summary artifact: `{artifact_id}`. Summary content: {result}"

    async def _llm_complete(
        self,
        # AliasChoices accommodates models that emit PascalCase or space-separated
        # parameter names instead of snake_case when invoking tool calls.
        answer: Annotated[
            str,
            Field(validation_alias=AliasChoices("answer", "Answer")),
        ],
        document_source_infos: Annotated[
            list[DocumentSourceInfo] | None,
            Field(
                validation_alias=AliasChoices(
                    "document_source_infos",
                    "Document Source Infos",
                    "DocumentSourceInfos",
                    "documentSources",
                )
            ),
        ] = None,
    ) -> DocumentAgentQueryResult:
        """Provide your final answer and document sources used.

        Parameters
        ----------
        answer : str
            Your final answer to the query.
        document_source_infos : list[DocumentSourceInfo] or None
            List of DocumentSourceInfo objects referencing content you used.
            Each should have:
            - content_id: The ID from search results or artifact IDs from summaries
            - relevant_direct_quotes: Optional list of verbatim quotes from the source

        Important
        ---------
        Use the exact snake_case parameter names above when calling this
        function, especially ``document_source_infos`` for citations.
        """
        artifacts = []
        citations = []
        document_source_infos = document_source_infos or []
        if not document_source_infos:
            answer = (
                "I could not find the requested information in the provided "
                f"document(s). {answer}"
            )

        self._logging_service.info(
            "Retrieving chunks and artifacts: %s", document_source_infos
        )
        for document_source_info in document_source_infos:
            if document_agent_artifact := self._artifacts.get(
                document_source_info.content_id
            ):
                self._logging_service.info(
                    "Appending artifact: %s", document_source_info.content_id
                )

                document = self._get_document_from_hash(
                    document_hash=document_agent_artifact.document_hash
                )

                if document is None:
                    raise FunctionCallError(
                        f"Unable to find file with document_hash: {document_agent_artifact.document_hash}"  # noqa: E501
                    )
                # Only create artifacts for structured data (tables), not text summaries
                # Text summaries should be streamed as regular responses, not artifacts
                if self._should_create_artifact(document_agent_artifact.content):
                    artifacts.append(
                        CopilotArtifact(
                            content=document_agent_artifact.content,
                            source_info=SourceInfo(
                                type="artifact",
                                origin="Chat",
                                uuid=uuid.uuid4(),
                                name=f"data-artifact-from-{document.filename}-{str(uuid.uuid4())[:5]}",
                                description=answer,
                            ),
                            data_format=RawObjectDataFormat(parse_as="table"),
                        )
                    )
                # For text summaries, we still create citations but no artifacts
                citations.append(
                    Citation(
                        source_info=document.source_info,
                        details=[
                            {
                                "Source type": document.source_info.type,
                                "Origin": document.source_info.origin,
                                "Data source": document.source_info.name,
                                "Name": document.source_info.name,
                                "Filename": document.filename,
                            }
                        ],
                        # We don't provide artifacts in citations if they've
                        # been explicitly produced by the agent.
                    )
                )

            # In the case below, we also don't provide an artifact as part of
            # the result (we only do it as part of the citation)
            elif retrieved_chunk := self._get_retrieved_chunk(
                retrieved_chunk_id=document_source_info.content_id
            ):
                self._logging_service.info(
                    "Appending search result: %s", document_source_info.content_id
                )
                document = self._get_document_from_hash(
                    document_hash=retrieved_chunk.metadata.get("document_hash")
                )

                if document is None:
                    raise FunctionCallError(
                        f"Unable to find content with content_id: {document_source_info.content_id}\n"  # noqa: E501
                        f"The choices are: {[chunk.retrieved_chunk_id for chunk in self._retrieved_document_chunks] + list(self._artifacts.keys())}"  # noqa: E501
                    )

                # Get bounding boxes for direct quotes (but only for PDFs!)
                if (
                    document_source_info.relevant_direct_quotes
                    and document.extension == "pdf"
                ):
                    direct_quotes_bounding_boxes = []
                    direct_quotes_bounding_boxes += (
                        self._get_direct_quotes_bounding_boxes(
                            document_source_info=document_source_info
                        )
                    )
                else:
                    direct_quotes_bounding_boxes = None

                if direct_quotes_bounding_boxes:
                    for direct_quote in direct_quotes_bounding_boxes:
                        citations.append(
                            Citation(
                                source_info=document.source_info,
                                quote_bounding_boxes=[direct_quote],
                                details=[
                                    {
                                        "Name": document.source_info.name,
                                        "Filename": document.filename,
                                        **(
                                            {
                                                "Page": retrieved_chunk.metadata.get(
                                                    "page_number"
                                                )
                                            }
                                            if "page_number" in retrieved_chunk.metadata
                                            else {}
                                        ),
                                    }
                                ],
                            )
                        )
                else:
                    citations.append(
                        Citation(
                            source_info=document.source_info,
                            details=[
                                {
                                    "Name": document.source_info.name,
                                    "Filename": document.filename,
                                    **(
                                        {
                                            "Page": retrieved_chunk.metadata.get(
                                                "page_number"
                                            )
                                        }
                                        if "page_number" in retrieved_chunk.metadata
                                        else {}
                                    ),
                                }
                            ],
                        )
                    )
            else:
                raise FunctionCallError(
                    f"Unable to find content: {document_source_info.content_id}. Are you sure you provided the right content_id?"  # noqa: E501
                )
        return DocumentAgentQueryResult(
            answer=answer,
            artifacts=artifacts,
            citations=citations,
        )

    def _should_create_artifact(self, content: str | list[dict]) -> bool:
        """Determine if content should be created as an artifact.

        Only creates artifacts for structured data with multiple rows and columns.
        Text summaries and single-row/single-column data are not converted to artifacts.

        Args:
            content: The content to check

        Returns:
            True if artifact should be created, False otherwise
        """
        # If content is a string, it's text - don't create artifact
        if isinstance(content, str):
            return False

        # If content is a list of dictionaries (table format)
        if isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict):
                # Only create artifact if we have multiple rows AND multiple columns
                return len(content) > 1 and len(content[0].keys()) > 1

        # Default to not creating artifact
        return False

    def _get_retrieved_chunk(
        self, retrieved_chunk_id: str
    ) -> RetrievedDocumentChunk | None:
        for chunk in self._retrieved_document_chunks:
            if str(chunk.retrieved_chunk_id) == str(retrieved_chunk_id):
                return chunk
        return None

    def _get_document_from_hash(self, document_hash: str | None) -> Document | None:
        for file_ in self._loaded_files:
            if file_.hash == document_hash:
                return file_
        return None

    def _get_target_words_from_document(
        self,
        document: Document,
        page_number: int | None,
        direct_quote: str,
    ) -> WordGroup | None:
        if not page_number:
            self._logging_service.warning(
                "No page number provided for direct quote: %s and file: %s. Using first page.",  # noqa: E501
                direct_quote,
                document.filename,
            )
            page_number = 1
        target_words = re.sub(r"[^\w\s]", "", direct_quote.lower()).split()

        with pdfplumber.open(io.BytesIO(document.content)) as pdf:
            # NB: PDF pages are 1-indexed, but our page_number is 0-indexed.
            page = pdf.pages[page_number - 1]
            extracted_words = page.extract_words()

            if not extracted_words:
                return None

            cleaned_words = [
                re.sub(r"[^\w\s]", "", word["text"].lower().strip())
                for word in extracted_words
            ]

            start_idx, end_idx = find_longest_common_substring_indices(
                document_words=cleaned_words, target_words=target_words
            )

            if end_idx - start_idx == 0:
                return None

            words = [Word(**word) for word in extracted_words]

            return WordGroup(
                page=page_number,
                words=words[start_idx:end_idx],
            )

    def _get_bounding_boxes_for_target_words(
        self, word_group: WordGroup | None
    ) -> list[CitationHighlightBoundingBox]:
        if word_group is None:
            return []

        groups: list[list[Word]] = []
        current_group: list[Word] = []
        for word in word_group.words:
            if not current_group:
                current_group.append(word)
            # If the word is on the same line as the current group, add it to
            # the group
            elif word.top == current_group[0].top:
                current_group.append(word)
            # Otherwise, start a new group
            else:
                groups.append(current_group)
                current_group = [word]
        groups.append(current_group)

        # Create the actual bounding boxes for each group.
        bboxes = []
        for group in groups:
            x0 = min(w.x0 for w in group)
            x1 = max(w.x1 for w in group)
            top = min(w.top for w in group)
            bottom = max(w.bottom for w in group)
            bbox = CitationHighlightBoundingBox(
                text=" ".join(w.text for w in group),
                page=word_group.page,
                x0=x0,
                top=top,
                x1=x1,
                bottom=bottom,
            )
            bboxes.append(bbox)

        return bboxes

    def _get_direct_quotes_bounding_boxes(
        self,
        document_source_info: DocumentSourceInfo,
    ) -> list[list[CitationHighlightBoundingBox]]:
        # TODO: We are technically retrieving these again (once in the outer loop
        # that cools this method, and then again here). The performance impact
        # is negligible, but it is a little untidy. See if we can clean it up in
        # future.
        bboxes = []
        if retrieved_chunk := self._get_retrieved_chunk(
            retrieved_chunk_id=document_source_info.content_id
        ):
            downloaded_user_file = self._get_document_from_hash(
                document_hash=retrieved_chunk.metadata.get("document_hash")
            )

            if downloaded_user_file is None:
                raise FunctionCallError(
                    f"Unable to find document with document_hash: {retrieved_chunk.metadata.get('document_hash')}"  # noqa: E501
                )

            self._logging_service.info(
                "Getting bounding boxes. File: %s. Quotes: %s",
                downloaded_user_file.filename,
                document_source_info.relevant_direct_quotes,
            )
            if not document_source_info.relevant_direct_quotes:
                return []
            for direct_quote in document_source_info.relevant_direct_quotes:
                word_group = self._get_target_words_from_document(
                    document=downloaded_user_file,
                    # We use the page number to speed up search times, otherwise we
                    # have to search the whole PDF, which can get slow for 30+ pages
                    # with lots of text.
                    page_number=retrieved_chunk.metadata.get("page_number"),
                    direct_quote=direct_quote,
                )
                if word_group:
                    bboxes.append(self._get_bounding_boxes_for_target_words(word_group))
        return bboxes
