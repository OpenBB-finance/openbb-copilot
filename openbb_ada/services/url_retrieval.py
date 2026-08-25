import asyncio
import logging
import os

import httpx
from async_lru import alru_cache
from tenacity import RetryError, retry, stop_after_attempt, wait_fixed

from .. import constants
from ..models import (
    Citation,
    SourceInfo,
    WebContext,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UrlRetrievalService:
    """Provide functionality for retrieving URLs."""

    @staticmethod  # <-- Important to prevent memory leaks on methods with an lru_cache
    @alru_cache(maxsize=1024, ttl=3600)
    async def retrieve_url(url: str) -> WebContext:
        # We have to use the tenacity retry decorator in isolation on a helper
        # function because it doesn't play nice with the lru_cache or
        # staticmethod decorators.
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
        async def _make_request(url: str) -> WebContext:
            async with httpx.AsyncClient(timeout=15) as client:
                if (jina_base_url := constants.JINA_AI_BASE_URL) is None:
                    raise ValueError("JINA_AI_BASE_URL is not set.")
                fetch_url = jina_base_url + "/" + url
                JINA_AI_API_KEY = os.getenv("JINA_AI_API_KEY")
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {JINA_AI_API_KEY}",
                }
                logger.debug("Retrieving URL: %s", fetch_url)
                response = await client.get(fetch_url, headers=headers)
                logger.info(
                    "Done retrieving URL: %s . Content: %s",
                    fetch_url,
                    response.text[:50],
                )
                web = WebContext(
                    url=url,
                    content=response.text,
                    citation=Citation(
                        source_info=SourceInfo(type="web", name=url),
                        details=[{"Website": url}],
                    ),
                )
                return web

        try:
            result = await _make_request(url=url)
            return result
        except RetryError:
            logger.info("Request timed out after last attempt: %s", url)
            return WebContext(url=url, content="Request timed out.")

    async def retrieve_urls(self, urls: list[str]) -> list[WebContext]:
        tasks = []
        for url in urls:
            tasks.append(UrlRetrievalService.retrieve_url(url))
        return await asyncio.gather(*tasks)
