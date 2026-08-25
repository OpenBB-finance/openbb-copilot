import asyncio
import tempfile
import zipfile
from pathlib import Path

import httpx

from .. import constants
from ..models import (
    Document,
    HttpMethod,
    SourceInfo,
    TargetUserFile,
    UnavailableDocument,
    UrlFileReference,
    UserFile,
)


class UserFileService:
    """Handle retrieving and uploading user files to the OpenBB Hub."""

    # TODO: This service can be removed, don't see it used anywhere
    # besides tests.

    def __init__(
        self,
        base_url: str,
        access_token: str,
        user_id: str,
    ):
        self._base_url = base_url
        self._access_token = access_token
        self._client = httpx.AsyncClient()
        self._headers = {
            "Authorization": f"Bearer {self._access_token}",
            "X-OpenBB-Authorization": f"Bearer {constants.OPENBB_PAYMENTS_API_SECRET_KEY}",  # noqa: E501
        }
        self._user_id = user_id

    async def _make_request(
        self,
        method: HttpMethod,
        path: str,
        payload: list | dict | None = None,
        files: list[tuple[str, tuple[str, bytes]]] | None = None,
    ) -> httpx.Response:
        response = await self._client.request(
            method,
            self._base_url + path,
            headers=self._headers,
            json=payload,
            files=files,
            timeout=15,
        )
        return response

    async def bulk_download_external_files(
        self, url_file_references: list[UrlFileReference]
    ) -> list[Document | UnavailableDocument]:
        responses = await asyncio.gather(
            *(
                self._client.get(str(url_file_reference.url), timeout=15)
                for url_file_reference in url_file_references
            )
        )
        documents: list[Document | UnavailableDocument] = []
        for response, url_file_reference in zip(
            responses, url_file_references, strict=True
        ):
            if response.status_code == 200 and response.content:
                documents.append(
                    Document(
                        content=response.content,
                        filename=url_file_reference.filename,
                        extension=url_file_reference.extension,
                        source_info=url_file_reference.source_info,
                    )
                )
            else:
                documents.append(
                    UnavailableDocument(
                        error=f"Failed to download file: {url_file_reference.url}, error: {response.text}.",  # noqa: E501
                        source_info=url_file_reference.source_info,
                    )
                )
        return documents

    async def upload_file(self, filename: str, content: str | bytes) -> UserFile:
        if isinstance(content, str):
            content = content.encode("utf-8")

        response = await self._make_request(
            HttpMethod.POST, "/pro/files", files=[("files", (filename, content))]
        )
        file_description = response.json()[0]
        return UserFile(
            file_uuid=file_description["stored_file_uuid"],
            filename=file_description["original_file_name"],
            extension=file_description["extension"],
        )

    async def bulk_upload_files(
        self, files: list[tuple[str, str | bytes]]
    ) -> list[UserFile]:
        files_payload = []
        for file_name, content in files:
            encoded_content = (
                content.encode("utf-8") if isinstance(content, str) else content
            )
            files_payload.append(("files", (file_name, encoded_content)))

        response = await self._make_request(
            HttpMethod.POST, "/pro/files", files=files_payload
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to upload files: {response.text}")

        user_files = []
        for file_description in response.json():
            user_files.append(
                UserFile(
                    file_uuid=file_description["stored_file_uuid"],
                    filename=file_description["original_file_name"],
                    extension=file_description["extension"],
                )
            )
        return user_files

    async def read_file(self, target_file: UserFile) -> Document:
        response = await self._make_request(
            HttpMethod.GET, f"/pro/files/{target_file.file_uuid}"
        )
        return Document(
            **target_file.model_dump(),
            content=response.content,
            source_info=SourceInfo(
                type="widget",
                uuid=target_file.file_uuid,
                name=target_file.filename,
            ),
        )

    async def bulk_download_files(
        self, target_files: list[TargetUserFile]
    ) -> list[Document]:
        payload = [
            {"file_uuid": str(target_file.file_uuid)} for target_file in target_files
        ]
        response = await self._make_request(
            HttpMethod.POST, "/pro/files/download", payload=payload
        )

        if response.status_code != 200:
            raise RuntimeError(f"Failed to download files: {response.text}")

        with tempfile.TemporaryDirectory() as temp_dir:
            zip_file_path = Path(temp_dir) / "files.zip"
            with open(zip_file_path, "wb") as f:
                f.write(response.content)
            with zipfile.ZipFile(zip_file_path, "r") as zip_file:
                zip_file.extractall(temp_dir)
            zip_file_path.unlink()

            downloaded_files: list[Document] = []
            for target_file in target_files:
                file_path = (
                    Path(temp_dir) / f"{target_file.file_uuid}.{target_file.extension}"
                )
                if file_path.exists():
                    downloaded_files.append(
                        Document(
                            **target_file.model_dump(),
                            content=file_path.read_bytes(),
                        )
                    )
            return downloaded_files

    async def delete_file(self, target_file: UserFile) -> bool:
        response = await self._make_request(
            HttpMethod.DELETE, f"/pro/files/{target_file.file_uuid}"
        )
        if "success" in response.json():
            return True
        else:
            return False

    async def close(self):
        await self._client.aclose()
