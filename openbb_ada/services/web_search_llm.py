import asyncio
from typing import (
    AsyncGenerator,
)

import openai
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_output_text import AnnotationURLCitation
from openbb_ai.models import (
    CitationCollection,
    CitationCollectionSSE,
    LlmClientMessage,
    MessageChunkSSE,
    MessageChunkSSEData,
    StatusUpdateSSE,
    StatusUpdateSSEData,
)

from ..constants import (
    OPENBB_SEARCH_API_KEY,
    OPENBB_SEARCH_BASE_URL,
    OPENBB_SEARCH_MODEL,
)
from ..models import (
    Citation,
    SourceInfo,
)
from ..utils.utils import (
    handle_openai_error,
    instrument_async_generator,
    sanitize_str,
)
from ._logging import LoggingService
from .template import TemplateService


class WebSearchLlmService:
    """Use an online LLM to perform web searches."""

    def __init__(
        self,
        logging_service: LoggingService,
        template_service: TemplateService,
        openai_api_key: str | None,
    ):
        self._logging_service = logging_service
        self._template_service = template_service
        self._openai_api_key = openai_api_key

    @instrument_async_generator("WebSearchLlmService.query")
    async def query(
        self, messages: list[LlmClientMessage], summary: str = "Searching the web"
    ) -> AsyncGenerator[
        StatusUpdateSSE | MessageChunkSSE | CitationCollectionSSE, None
    ]:
        self._logging_service.info("Web search flow triggered: %s", summary)
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
                artifacts=[],
            )
        )

        # Need to have a slightly delay for the SSE to send immediately after
        # the connection is established.
        await asyncio.sleep(0.1)
        chat_messages: list[dict] = [
            {
                "role": "developer",
                "content": "You are a helpful assistant that can search the web for information.",  # noqa: E501
            }
        ]

        summary_message_str = (
            "The following is a summary of the conversation thus far:\n\n"
        )
        for message in messages[:-1]:
            if message.role == "ai":
                summary_message_str += sanitize_str(f"AI: {message.content}\n\n")
            elif message.role == "tool":
                pass
            elif message.role == "human":
                summary_message_str += sanitize_str(f"USER: {message.content}\n\n")

        chat_messages.extend(
            [
                {
                    "role": "user",
                    "content": sanitize_str(summary_message_str),
                },
                {
                    "role": "assistant",
                    "content": "Understood.",
                },
                {
                    "role": "user",
                    "content": sanitize_str(str(messages[-1].content)),
                },
            ]
        )

        client = openai.AsyncOpenAI(
            base_url=OPENBB_SEARCH_BASE_URL,
            api_key=self._openai_api_key or OPENBB_SEARCH_API_KEY,
        )
        try:
            response_stream = await client.responses.create(
                model=OPENBB_SEARCH_MODEL,
                tools=[{"type": "web_search_preview"}],
                input=chat_messages,  # type: ignore[arg-type]
                stream=True,
            )

            citations: set[Citation] = set()
            async for event in response_stream:  # type: ignore[union-attr]
                if isinstance(event, ResponseTextDeltaEvent):
                    yield MessageChunkSSE(
                        data=MessageChunkSSEData(
                            delta=event.delta,
                        )
                    )
                elif isinstance(event, ResponseCompletedEvent):
                    for output in event.response.output:
                        if isinstance(output, ResponseOutputMessage):
                            for output_content in output.content:
                                if isinstance(output_content, ResponseOutputText):
                                    for annotation in output_content.annotations:
                                        if isinstance(
                                            annotation, AnnotationURLCitation
                                        ):
                                            citations.add(
                                                Citation(
                                                    source_info=SourceInfo(
                                                        type="web",
                                                        name=annotation.title,
                                                    ),
                                                    details=[
                                                        {
                                                            "Link": annotation.url,
                                                            "Title": annotation.title,
                                                        }
                                                    ],
                                                )
                                            )
            if citations:
                yield CitationCollectionSSE(
                    data=CitationCollection(citations=list(citations))
                )
            return
        except openai.OpenAIError as err:
            self._logging_service.error("LLM call failed: %s", err)
            yield handle_openai_error(err)
            return
