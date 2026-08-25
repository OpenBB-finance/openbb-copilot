import asyncio
import io
import tempfile
import zipfile
from pathlib import Path
from typing import (
    AsyncGenerator,
)

import pandas as pd
import redis.asyncio as redis
from openbb_ai.models import (
    StatusUpdateSSE,
    StatusUpdateSSEData,
)

from .. import constants
from ..models import (
    Citation,
    DataSource,
    Document,
    DocumentAgentQueryResult,
    DocumentQueryResult,
    SqlAgentQueryResult,
    SqlTableInfo,
    UnavailableDocument,
    UserFile,
)
from ..utils.ai import get_llm
from ..vector_db import VectorDb
from ._logging import LoggingService
from .document_agent import DocumentAgentService
from .sql_agent import SqlAgentService
from .template import TemplateService
from .user_file import UserFileService


class DocumentService:
    """Handle parsing and querying user-uploaded documents."""

    STRUCTURED_FILE_EXTENSIONS = constants.STRUCTURED_FILE_EXTENSIONS
    IMAGE_FILE_EXTENSIONS = constants.IMAGE_FILE_EXTENSIONS
    UNSTRUCTURED_FILE_EXTENSIONS = constants.UNSTRUCTURED_FILE_EXTENSIONS
    SUPPORTED_FILE_EXTENSIONS = constants.SUPPORTED_FILE_EXTENSIONS

    def __init__(
        self,
        user_file_service: UserFileService,
        template_service: TemplateService,
        document_agent_service: DocumentAgentService,
        sql_agent_service: SqlAgentService,
        logging_service: LoggingService,
        documents: list[Document | UnavailableDocument],
        openai_api_key: str | None,
    ):
        self._user_file_service = user_file_service
        self._template_service = template_service
        self._document_agent_service = document_agent_service
        self._sql_agent_service = sql_agent_service
        self._logging_service = logging_service
        self._documents: list[Document] = [
            x for x in documents if isinstance(x, Document)
        ]
        self._unavailable_documents: list[UnavailableDocument] = [
            x for x in documents if isinstance(x, UnavailableDocument)
        ]
        self._all_user_files: list[UserFile] = []
        self._sql_table_to_filename_map: dict[str, str] = {}
        self._base_dir = Path(tempfile.mkdtemp())
        self._openai_api_key = openai_api_key

    def get_document_by_hash(self, xxh64_hash: str) -> Document | None:
        for document in self._documents:
            if document.hash == xxh64_hash:
                return document
        return None

    def get_unavailable_documents_by_data_source(
        self, data_source: DataSource
    ) -> list[UnavailableDocument]:
        unavailable_documents: list[UnavailableDocument] = []
        for unavailable_document in self._unavailable_documents:
            if (
                unavailable_document.source_info.origin == data_source.origin
                and unavailable_document.source_info.widget_id == data_source.id
            ):
                unavailable_documents.append(unavailable_document)
        return unavailable_documents

    async def _load_documents(self, documents: list[Document]) -> list[VectorDb]:
        tasks = []
        for doc in documents:
            tasks.append(
                self._document_agent_service.load_unstructured_document(document=doc)
            )
        vector_dbs = await asyncio.gather(*tasks)
        return vector_dbs

    @staticmethod
    def _get_aredis_client() -> redis.Redis:
        connection_pool = redis.ConnectionPool(
            host=constants.REDIS_HOST,
            port=constants.REDIS_PORT,
            db=0,
        )
        redis_client = redis.Redis(connection_pool=connection_pool)
        return redis_client

    def _get_document_cache_key(self, document: Document) -> str:
        return f"vector_db:{self._user_file_service._user_id}:{document.hash}"

    @staticmethod
    def _extract_and_load_downloaded_vector_db(vector_db_zip: bytes) -> VectorDb:
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer = io.BytesIO(vector_db_zip)
            with zipfile.ZipFile(buffer, "r") as zf:
                zf.extractall(tmpdir)
            return VectorDb(load_from_path=tmpdir)

    async def _store_vector_dbs(
        self, doc_vector_db_pairs: list[tuple[Document, VectorDb]]
    ) -> None:
        async with self._get_aredis_client() as aredis_client:

            async def store_document_vector_db(
                document: Document, vector_db: VectorDb
            ) -> None:
                try:
                    self._logging_service.info(
                        "Storing vector database for file: %s", document.filename
                    )
                    buffer = io.BytesIO()
                    vector_db.save_to_buffer(buffer)
                    key = self._get_document_cache_key(document)
                    KEY_EXPIRATION = 172800  # 2 days
                    await aredis_client.set(key, buffer.getvalue(), KEY_EXPIRATION)
                except redis.ConnectionError as e:
                    self._logging_service.error(
                        "Unable to connect to document cache: %s. Proceeding without cache.",  # noqa: E501
                        e,
                    )
                except Exception as e:
                    self._logging_service.error(
                        "Failed to store vector database for file: %s",
                        document.filename,
                    )
                    raise e

            tasks = [
                store_document_vector_db(doc, vector_db)
                for doc, vector_db in doc_vector_db_pairs
            ]
            await asyncio.gather(*tasks)

    async def _vector_db_exists(self, document: Document) -> bool:
        async with self._get_aredis_client() as aredis_client:
            try:
                key = self._get_document_cache_key(document)
                exists = await aredis_client.exists(key)
                return bool(exists)
            except redis.ConnectionError as e:
                self._logging_service.error(
                    "Unable to connect to document cache: %s. Proceeding without cache.",  # noqa: E501
                    e,
                )
                return False
            except Exception as e:
                self._logging_service.error(
                    "Failed to check if vector database exists for file: %s",
                    document.filename,
                )
                raise e

    async def _load_vector_dbs(
        self, documents: list[Document]
    ) -> list[tuple[Document, VectorDb]]:
        async with self._get_aredis_client() as aredis_client:

            async def load_document_vector_db(
                document: Document,
            ) -> tuple[Document, VectorDb] | None:
                try:
                    key = self._get_document_cache_key(document)
                    if vector_db_zip := await aredis_client.get(key):
                        db = await asyncio.to_thread(
                            DocumentService._extract_and_load_downloaded_vector_db,
                            vector_db_zip,
                        )
                        return document, db
                    return None
                except Exception as e:
                    self._logging_service.error(
                        "Failed to load vector database for file: %s", document.filename
                    )
                    raise e

            tasks = [load_document_vector_db(doc) for doc in documents]
            doc_vector_db_pairs = await asyncio.gather(*tasks)
            return [pair for pair in doc_vector_db_pairs if pair]

    async def _init(self) -> None:
        # Load structured documents into SQLAgent
        structured_documents = [
            doc for doc in self._documents if self._is_file_structured(doc)
        ]
        if structured_documents:
            self._logging_service.info("Loading structured files into SQLite...")
            for doc in structured_documents:
                await self.load_spreadsheet_like(doc)

        # Load unstructured documents into DocumentAgent
        # First start with images
        if self.user_has_image_data():
            images_files = [doc for doc in self._documents if self._is_file_image(doc)]
            self._logging_service.info(
                "Loading image files into DocumentAgent: %s",
                [doc.filename for doc in images_files],
            )
            for doc in images_files:
                await self._document_agent_service.load_image(doc)

        # Next, load the rest of the unstructured files
        docs_requiring_vector_db: list[Document] = [
            doc for doc in self._documents if self._is_file_unstructured(doc)
        ]
        if docs_requiring_vector_db:
            if not constants.USE_DOCUMENT_CACHE:
                filenames = [doc.filename for doc in docs_requiring_vector_db]
                self._logging_service.info(
                    "Cache disabled - creating new vector databases for all files: %s",  # noqa: E501
                    filenames,
                )
                tasks = [
                    self._document_agent_service.load_unstructured_document(doc)
                    for doc in docs_requiring_vector_db
                ]
                await asyncio.gather(*tasks)
            else:
                docs_with_db: dict[str, Document] = {}
                docs_missing_db: dict[str, Document] = {}
                for doc in docs_requiring_vector_db:
                    if await self._vector_db_exists(doc):
                        docs_with_db[doc.filename] = doc
                    else:
                        docs_missing_db[doc.filename] = doc

                if docs_with_db:
                    filenames = list(docs_with_db.keys())
                    documents = list(docs_with_db.values())
                    self._logging_service.info(
                        "Loading existing vector databases for files: %s", filenames
                    )
                    existing_doc_vector_db_pairs = await self._load_vector_dbs(
                        documents
                    )
                    tasks = [
                        self._document_agent_service.load_unstructured_document(
                            doc, vector_db
                        )
                        for doc, vector_db in existing_doc_vector_db_pairs
                    ]
                    await asyncio.gather(*tasks)

                if docs_missing_db:
                    filenames = list(docs_missing_db.keys())
                    documents = list(docs_missing_db.values())
                    self._logging_service.info(
                        "Creating new vector databases for files: %s", filenames
                    )
                    vector_dbs_to_store: list[VectorDb] = []
                    tasks = [
                        self._document_agent_service.load_unstructured_document(doc)
                        for doc in documents
                    ]
                    vector_dbs_to_store += await asyncio.gather(*tasks)
                    self._logging_service.info(
                        "Storing new vector databases for files: %s", filenames
                    )
                    if new_doc_vector_db_pairs := list(
                        zip(documents, vector_dbs_to_store, strict=True)
                    ):
                        await self._store_vector_dbs(new_doc_vector_db_pairs)

    def _get_model(self, **kwargs):
        return get_llm(temperature=0.1, api_key=self._openai_api_key, **kwargs)

    async def query_unstructured_files(
        self, query: str
    ) -> AsyncGenerator[StatusUpdateSSE | DocumentQueryResult, None]:
        document_agent_query_result: DocumentAgentQueryResult = (
            await self._document_agent_service.query(query=query)
        )
        # Yield the status update SSEs
        if document_agent_query_result.artifacts:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message="Artifact generated",
                    details=[],
                    artifacts=[
                        artifact.to_client_artifact()
                        for artifact in document_agent_query_result.artifacts
                    ],
                )
            )
        else:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message="Accessing files",
                    details=[
                        {
                            "name": citation.details[0]["Name"],
                            "filename": citation.details[0]["Filename"],
                            **(
                                {"page": citation.details[0]["Page"]}
                                if citation.details[0].get("Page")
                                else {}
                            ),
                        }
                        for citation in document_agent_query_result.citations
                        if citation.details
                    ],
                )
            )

        # Yield the final result
        yield DocumentQueryResult(
            answer=document_agent_query_result.answer,
            citations=document_agent_query_result.citations,
            artifacts=document_agent_query_result.artifacts,
        )

    def _get_downloaded_user_file_by_filename(self, filename: str) -> Document:
        for downloaded_user_file in self._documents:
            if downloaded_user_file.filename == filename:
                return downloaded_user_file
        raise ValueError(f"Unable to find file: {filename}")

    async def query_structured_files(
        self, query: str
    ) -> AsyncGenerator[DocumentQueryResult | StatusUpdateSSE, None]:
        result = self._sql_agent_service.query(query)
        sql_agent_query_result: SqlAgentQueryResult | None = None
        async for event in result:
            if isinstance(event, StatusUpdateSSE):
                yield event
            elif isinstance(event, SqlAgentQueryResult):
                sql_agent_query_result = event
                break

        if sql_agent_query_result is None:
            return

        source_infos = []
        for table in sql_agent_query_result.queried_tables:
            if filename := self._sql_table_to_filename_map.get(table):
                source_infos.append(
                    self._get_downloaded_user_file_by_filename(filename).source_info
                )

        artifacts = (
            [sql_agent_query_result.artifact]
            if sql_agent_query_result.artifact
            else None
        )

        details = []
        for source_info in source_infos:
            detail = {**source_info.metadata} if source_info.metadata else {}
            detail["name"] = source_info.name
            details.append(detail)

        if artifacts:
            # We drop input args from the details for status updates that
            # contain artifacts (otherwise it's confusing to the user, since the
            # input args aren't part of the artifact, but the original source)
            details = [
                {k: v for k, v in detail.items() if k != "input_args"}
                for detail in details
            ]
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message="Artifact generated",
                    details=[],
                    artifacts=[
                        artifact.to_client_artifact()
                        for artifact in artifacts
                        if artifacts
                    ],
                )
            )

        yield DocumentQueryResult(
            answer=sql_agent_query_result.answer,
            citations=[
                Citation(
                    source_info=source_info,
                    details=[
                        {
                            "Name": source_info.name,
                            "Filename": source_info.metadata.get("filename"),
                        }
                    ],
                )
                for source_info in source_infos
            ],
            artifacts=artifacts,
        )

    async def query_all_user_files(
        self, query: str
    ) -> AsyncGenerator[list[DocumentQueryResult] | StatusUpdateSSE, None]:
        results = []
        if self.user_has_unstructured_data() or self.user_has_image_data():
            self._logging_service.info(
                "User has unstructured or image files. Querying: %s", query
            )
            async for event in self.query_unstructured_files(query):
                if isinstance(event, StatusUpdateSSE):
                    yield event
                elif isinstance(event, DocumentQueryResult):
                    results.append(event)

        if self.user_has_structured_data():
            self._logging_service.info("User has structured data. Querying: %s", query)
            structured_data_result = self.query_structured_files(query)
            async for event in structured_data_result:
                if isinstance(event, StatusUpdateSSE):
                    yield event
                elif isinstance(event, DocumentQueryResult):
                    results.append(event)

        yield results

    @staticmethod
    def _is_file_unstructured(user_file: Document) -> bool:
        return user_file.extension in DocumentService.UNSTRUCTURED_FILE_EXTENSIONS

    @staticmethod
    def _is_file_structured(user_file: Document) -> bool:
        return user_file.extension in DocumentService.STRUCTURED_FILE_EXTENSIONS

    @staticmethod
    def _is_file_image(user_file: Document) -> bool:
        return user_file.extension in DocumentService.IMAGE_FILE_EXTENSIONS

    @staticmethod
    def _is_file_supported(user_file: UserFile) -> bool:
        return user_file.extension in DocumentService.SUPPORTED_FILE_EXTENSIONS

    def _user_has_documents_with_extension(self, extensions: list[str]) -> bool:
        return any(document.extension in extensions for document in self._documents)

    def user_has_image_data(self) -> bool:
        return self._user_has_documents_with_extension(self.IMAGE_FILE_EXTENSIONS)

    def user_has_unstructured_data(self) -> bool:
        return self._user_has_documents_with_extension(
            self.UNSTRUCTURED_FILE_EXTENSIONS
        )

    def user_has_structured_data(self) -> bool:
        return self._user_has_documents_with_extension(self.STRUCTURED_FILE_EXTENSIONS)

    async def load_spreadsheet_like(self, downloaded_user_file: Document) -> None:
        self._logging_service.info(
            "Loading as structured document: %s", downloaded_user_file.filename
        )
        if downloaded_user_file.extension == "csv":
            df = pd.read_csv(
                io.BytesIO(downloaded_user_file.content), index_col=0, parse_dates=True
            )
            name: str = str(downloaded_user_file.filename)
            sql_table_info = await self._load_tabular_document(
                filename_or_sheet_name=name, df=df
            )
            self._update_sql_table_to_filename_map(
                table_name=sql_table_info.table_name,
                filename=downloaded_user_file.filename,
            )

        elif downloaded_user_file.extension == "xlsx":
            sheets = pd.read_excel(
                io.BytesIO(downloaded_user_file.content),
                index_col=0,
                parse_dates=True,
                sheet_name=None,
            )
            for sheet_name, df in sheets.items():
                sql_table_info = await self._load_tabular_document(
                    filename_or_sheet_name=sheet_name, df=df
                )
                self._update_sql_table_to_filename_map(
                    table_name=sql_table_info.table_name,
                    filename=downloaded_user_file.filename,
                )
        else:
            raise IOError(
                f"Tried loading a spreadsheet-like file, but it's not a CSV or XLSX file: {downloaded_user_file.filename}"  # noqa: E501
            )

    async def _load_tabular_document(
        self, filename_or_sheet_name: str, df: pd.DataFrame
    ) -> SqlTableInfo:
        table_name = self._derive_sql_table_name(filename_or_sheet_name)
        sql_table_info = self._sql_agent_service.insert_table(
            df=df,
            table_name=table_name,
            metadata={"source_filename_or_sheet_name": filename_or_sheet_name},
        )
        return sql_table_info

    @classmethod
    def _derive_sql_table_name(cls, filename_or_sheet_name: str) -> str:
        source_path = Path(filename_or_sheet_name)
        suffix = source_path.suffix.removeprefix(".").lower()
        if suffix in cls.STRUCTURED_FILE_EXTENSIONS:
            return source_path.stem
        return filename_or_sheet_name

    def _update_sql_table_to_filename_map(self, table_name: str, filename: str):
        self._sql_table_to_filename_map[table_name] = filename
        self._logging_service.info(
            "Updated SQL table map: %s", self._sql_table_to_filename_map
        )

    @property
    def all_user_files(self) -> list[UserFile]:
        return self._all_user_files

    @property
    def documents(self) -> list[Document]:
        return self._documents

    @property
    def downloaded_user_files(self) -> list[Document]:
        return self._documents

    @property
    def loaded_unstructured_files(self) -> list[Document]:
        return self._document_agent_service.loaded_files
