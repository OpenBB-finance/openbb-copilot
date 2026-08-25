import io
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from openbb_ada.vector_db import (
    VectorDb,
    VectorDbDocument,
    _chunk_texts,
    _OpenAIEmbeddingProvider,
    _SnowflakeEmbeddingProvider,
)


class DummyProvider:
    def __init__(self, dimension: int = 1):
        self.dimension = dimension
        self.max_batch_tokens: int | None = None
        self.max_batch_size: int | None = None
        self.max_chars_per_item: int | None = None

    async def embed(  # noqa: E501
        self, texts: list[str], *, pooling: bool = True
    ) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            first_value = float(len(text) % 10)
            vector = [first_value] + [0.0] * (self.dimension - 1)
            embeddings.append(vector)
        return embeddings

    def count_tokens(self, text: str) -> int:
        return len(text)


@pytest.fixture
def make_dummy_provider():
    def _make(dimension: int = 1) -> DummyProvider:
        return DummyProvider(dimension)

    return _make


@pytest.mark.parametrize(
    ("provider", "default_model"),
    [
        ("openai", "text-embedding-3-small"),
        ("snowflake", "snowflake-arctic-embed-m-v1.5"),
    ],
)
def test_faiss_init_with_default_model(monkeypatch, provider, default_model):
    monkeypatch.setattr(
        "openbb_ada.vector_db.OPENBB_EMBEDDING_MODEL_PROVIDER",
        provider,
        raising=False,
    )
    monkeypatch.setattr(
        "openbb_ada.vector_db.OPENBB_EMBEDDING_MODEL",
        default_model,
        raising=False,
    )

    # Set SNOWFLAKE_HOST environment variable for snowflake provider
    if provider == "snowflake":
        monkeypatch.setenv("SNOWFLAKE_HOST", "test.snowflakecomputing.com")

    faiss_index = VectorDb()
    assert faiss_index.model == default_model
    assert faiss_index._dimensions is None
    assert faiss_index._index is None


@pytest.mark.parametrize(
    ("provider", "custom_model"),
    [
        ("openai", "text-embedding-ada-002"),
        ("snowflake", "snowflake-arctic-embed-l-v2.0"),
    ],
)
def test_faiss_init_with_custom_model(monkeypatch, provider, custom_model):
    monkeypatch.setattr(
        "openbb_ada.vector_db.OPENBB_EMBEDDING_MODEL_PROVIDER",
        provider,
        raising=False,
    )
    monkeypatch.setattr(
        "openbb_ada.vector_db.OPENBB_EMBEDDING_MODEL",
        custom_model,
        raising=False,
    )

    # Set SNOWFLAKE_HOST environment variable for snowflake provider
    if provider == "snowflake":
        monkeypatch.setenv("SNOWFLAKE_HOST", "test.snowflakecomputing.com")

    faiss_index = VectorDb()
    assert faiss_index.model == custom_model
    assert faiss_index._dimensions is None
    assert faiss_index._index is None


@pytest.mark.asyncio
async def test_faiss_add_documents(make_dummy_provider):
    faiss_index = VectorDb()
    faiss_index._provider = make_dummy_provider()
    documents = [
        VectorDbDocument(page_content="test page content", metadata={"a": "b"}),
        VectorDbDocument(page_content="test page content 2", metadata={"a": "b"}),
    ]
    await faiss_index.add(documents)
    assert faiss_index._index.ntotal == 2
    assert len(faiss_index._docs) == 2


@pytest.mark.asyncio
async def test_faiss_add_documents_with_empty_list_raises_error():
    faiss_index = VectorDb()
    documents = []
    with pytest.raises(ValueError):
        await faiss_index.add(documents)


def test_faiss_load_from_disk():
    test_path = Path(__file__).parent / "test_data" / "test_faiss_index"
    faiss_index = VectorDb()
    faiss_index.load_from_disk(test_path)
    assert faiss_index._index.ntotal == 5
    assert len(faiss_index._docs) == 5
    assert isinstance(faiss_index._docs[0], VectorDbDocument)
    assert isinstance(faiss_index._docs[1], VectorDbDocument)


@pytest.mark.asyncio
async def test_faiss_search(make_dummy_provider):
    test_path = Path(__file__).parent / "test_data" / "test_faiss_index"
    faiss_index = VectorDb()
    faiss_index.load_from_disk(test_path)
    faiss_index._provider = make_dummy_provider(faiss_index._index.d)
    results = await faiss_index.search("the meaning of life", k=3)
    assert len(results) == 3
    assert results[0].page_content == "So long, and thanks for all the fish."


@pytest.mark.asyncio
async def test_faiss_search_with_empty_index_raises_error():
    faiss_index = VectorDb()
    with pytest.raises(ValueError):
        await faiss_index.search("the meaning of life", k=3)


@pytest.mark.asyncio
async def test_faiss_search_doesnt_return_more_results_than_documents(
    make_dummy_provider,
):
    faiss_index = VectorDb()
    faiss_index._provider = make_dummy_provider()

    test_documents = [
        VectorDbDocument(page_content="first piece of content"),
        VectorDbDocument(page_content="second piece of content"),
        VectorDbDocument(page_content="third piece of content"),
    ]
    await faiss_index.add(test_documents)
    results = await faiss_index.search("first piece of content", k=10)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_faiss_save_to_disk(vector_db_target_path, make_dummy_provider):
    documents = [
        VectorDbDocument(
            page_content="The answer to the ultimate question of life, the universe, and everything is 42.",  # noqa: E501
            metadata={"a": "b"},
        ),
        VectorDbDocument(page_content="Don't Panic.", metadata={"a": "b"}),
        VectorDbDocument(
            page_content="So long, and thanks for all the fish.", metadata={"a": "b"}
        ),
        VectorDbDocument(
            page_content="Time is an illusion. Lunchtime doubly so.",
            metadata={"a": "b"},
        ),
        VectorDbDocument(
            page_content="I love deadlines. I love the whooshing noise they make as they go by.",  # noqa: E501
            metadata={"a": "b"},
        ),
    ]
    faiss_index = VectorDb()
    faiss_index._provider = make_dummy_provider()
    await faiss_index.add(documents)
    faiss_index.save_to_disk(vector_db_target_path)

    assert vector_db_target_path.exists()
    assert vector_db_target_path.is_dir()
    assert vector_db_target_path.name == vector_db_target_path.name
    assert (vector_db_target_path / "vector_index.index").exists()
    assert (vector_db_target_path / "vector_index.index").is_file()
    assert (vector_db_target_path / "documents.json").exists()
    assert (vector_db_target_path / "documents.json").is_file()


@pytest.mark.asyncio
async def test_faiss_save_to_buffer(make_dummy_provider):
    documents = [
        VectorDbDocument(
            page_content="The answer to the ultimate question of life, the universe, and everything is 42.",  # noqa: E501
            metadata={"a": "b"},
        ),
        VectorDbDocument(page_content="Don't Panic.", metadata={"a": "b"}),
        VectorDbDocument(
            page_content="So long, and thanks for all the fish.", metadata={"a": "b"}
        ),
        VectorDbDocument(
            page_content="Time is an illusion. Lunchtime doubly so.",
            metadata={"a": "b"},
        ),
        VectorDbDocument(
            page_content="I love deadlines. I love the whooshing noise they make as they go by.",  # noqa: E501
            metadata={"a": "b"},
        ),
    ]
    faiss_index = VectorDb()
    faiss_index._provider = make_dummy_provider()
    await faiss_index.add(documents)
    test_buffer = io.BytesIO()
    test_buffer = faiss_index.save_to_buffer(test_buffer)

    # Save buffer to disk, and unzip, then load the vectordb to test.
    with tempfile.NamedTemporaryFile(suffix=".zip") as temp_file:
        temp_file.write(test_buffer.getvalue())
        temp_file.flush()

        # Unzip
        extract_dir = temp_file.name.rstrip(".zip")
        shutil.unpack_archive(temp_file.name, extract_dir=extract_dir)

    loaded_faiss_index = VectorDb()
    loaded_faiss_index.load_from_disk(extract_dir)
    loaded_faiss_index._provider = make_dummy_provider(loaded_faiss_index._index.d)

    assert len(loaded_faiss_index.docs) == 5
    assert len(await loaded_faiss_index.search("the meaning of life", k=3)) == 3


def test_chunk_texts_respects_limits():
    class LimitedProvider(DummyProvider):
        def __init__(self):
            super().__init__(dimension=1)
            self.max_batch_tokens = 5
            self.max_batch_size = 2

    provider = LimitedProvider()
    batches = _chunk_texts(["aa", "bbb", "cccc"], provider)
    assert batches == [["aa"], ["bbb"], ["cccc"]]


@pytest.mark.asyncio
async def test_openai_embedding_provider_embed(monkeypatch):
    provider = _OpenAIEmbeddingProvider(
        provider="openai",
        api_key="test-key",
        base_url="https://example.com",
    )

    async def fake_create(*args, **kwargs):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])

    monkeypatch.setattr(provider._client.embeddings, "create", fake_create)

    embeddings = await provider.embed(["hello"])
    assert embeddings == [[0.1, 0.2, 0.3]]
    assert provider.count_tokens("hello") >= 0


@pytest.mark.asyncio
async def test_snowflake_embedding_provider_embed(monkeypatch):
    provider = _SnowflakeEmbeddingProvider(
        api_key="pat-token",
        base_url="https://example.snowflakecomputing.com",
    )

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"embedding": [1, 2]}, {"embedding": [3, 4]}]}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("openbb_ada.vector_db.httpx.AsyncClient", FakeAsyncClient)

    embeddings = await provider.embed(["one", "two"])
    assert embeddings == [[0.4472135954999579, 0.8944271909999159], [0.6, 0.8]]
    assert provider.count_tokens("tokenize me") >= 0


@pytest.mark.asyncio
async def test_snowflake_embedding_provider_embed_long_page_pooled(monkeypatch):
    provider = _SnowflakeEmbeddingProvider(
        api_key="pat-token",
        base_url="https://example.snowflakecomputing.com",
    )

    # Build a 4300-char page that forces multiple sub-chunks under the provider's
    # splitter (max_chars_per_item=2000, target=1600, overlap=200)
    long_text = ("A" * 3299) + ". " + ("B" * 998)  # total length: 4300

    # Fixed embeddings to cycle through for each sub-chunk
    fixed_embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]

    class FakeResponse:
        def __init__(self, count):
            self._count = count

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"embedding": fixed_embeddings[i % len(fixed_embeddings)]}
                    for i in range(self._count)
                ]
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            batch = kwargs.get("json", {}).get("text", [])
            return FakeResponse(len(batch))

    monkeypatch.setattr("openbb_ada.vector_db.httpx.AsyncClient", FakeAsyncClient)

    # Run embedding – provider should split into sub-chunks, then pool to 1 vector
    embeddings = await provider.embed([long_text])
    assert len(embeddings) == 1
    vec = embeddings[0]
    assert len(vec) == 2

    import math

    # Unit length (L2-normalized)
    norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    assert norm == pytest.approx(1.0, rel=1e-6, abs=1e-9)
