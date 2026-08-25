import pytest

from openbb_ada.utils.stream import stream_message_chunks


@pytest.mark.asyncio
async def test_stream_message_chunks():
    """Test streaming message chunks."""
    chunks = []

    async for chunk in stream_message_chunks("hi"):
        chunks.append(chunk)

    # Should have 2 chunks for "hi"
    assert len(chunks) == 2

    # Check the chunks are MessageChunkSSE with correct deltas
    assert chunks[0].data.delta == "h"
    assert chunks[1].data.delta == "i"

    # Verify the type
    from openbb_ai.models import MessageChunkSSE

    assert all(isinstance(chunk, MessageChunkSSE) for chunk in chunks)
