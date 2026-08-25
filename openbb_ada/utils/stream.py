import asyncio
from typing import AsyncGenerator

from openbb_ai.models import (
    MessageChunkSSE,
    MessageChunkSSEData,
)

MESSAGE_CHUNK_DELAY = 0.002


async def stream_message_chunks(chunk: str) -> AsyncGenerator[MessageChunkSSE, None]:
    for letter in chunk:
        # Smooth streaming
        await asyncio.sleep(MESSAGE_CHUNK_DELAY)
        yield MessageChunkSSE(data=MessageChunkSSEData(delta=letter))
