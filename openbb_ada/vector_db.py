import asyncio
import io
import json
import logging
import tempfile
import zipfile
from pathlib import Path

import faiss
import httpx
import numpy as np
import tiktoken
from fastapi.encoders import jsonable_encoder
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel, Field

from .constants import (
    OPENBB_EMBEDDING_API_KEY,
    OPENBB_EMBEDDING_BASE_URL,
    OPENBB_EMBEDDING_MODEL,
    OPENBB_EMBEDDING_MODEL_PROVIDER,
)

logger = logging.getLogger(__name__)


class _BaseEmbeddingProvider:
    max_batch_tokens: int | None = None
    max_batch_size: int | None = None
    max_chars_per_item: int | None = None
    default_model: str | None = None
    can_split_items: bool = False

    def __init__(self):
        if self.default_model is None:
            raise ValueError("Embedding model must be provided")
        self._model = self.default_model
        self._encoder = tiktoken.get_encoding("cl100k_base")

    async def embed(
        self, texts: list[str], *, pooling: bool = True
    ) -> list[list[float]]:
        """Embed a list of texts.

        Parameters
        ----------
        texts : list[str]
            The texts to embed.
        pooling : bool, optional
            If True (default), chunk long texts and pool embeddings.
            If False, truncate long texts and log a warning.
        """
        raise NotImplementedError

    def count_tokens(self, text: str) -> int | None:
        return None


class _OpenAIEmbeddingProvider(_BaseEmbeddingProvider):
    max_batch_tokens = int(300_000 * 0.9)
    max_batch_size = 2048
    default_model = "text-embedding-3-small"

    def __init__(
        self,
        provider: str,
        *,
        api_key: str | None,
        base_url: str | None,
        model: str | None = None,
    ):
        super().__init__()
        if model:
            self._model = model
        if provider == "openai":
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        elif provider == "azure":
            self._client = AsyncAzureOpenAI()
        else:
            raise ValueError("Unsupported OpenAI provider")

    async def embed(
        self, texts: list[str], *, pooling: bool = True
    ) -> list[list[float]]:
        # OpenAI has 8191 token limit, pooling not needed for most use cases
        response = await self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    def count_tokens(self, text: str) -> int | None:
        if self._encoder is None:
            return None
        return len(self._encoder.encode(text))


class _SnowflakeEmbeddingProvider(_BaseEmbeddingProvider):
    max_batch_size = 1280
    # Snowflake models have 512 token limit (~2000 chars)
    max_chars_per_item = 2000
    default_model = "snowflake-arctic-embed-m-v1.5"
    can_split_items = True

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        model: str | None = None,
    ):
        super().__init__()
        if model:
            self._model = model
        if not base_url or not api_key:
            raise ValueError("Snowflake embedding configuration is incomplete")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._encoder: tiktoken.Encoding | None = None
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = None

    async def embed(
        self, texts: list[str], *, pooling: bool = True
    ) -> list[list[float]]:
        # Provider-local micro-chunking and pooling to keep one vector per input page
        # Consecutive spaces are preserved (we do not collapse whitespace), because
        # some PDF table layouts depend on spacing.
        #
        # Strategy when pooling=True:
        # - For each input text, if len(text) > self.max_chars_per_item, split into
        #   overlapping sub-chunks (target ~1600 chars, overlap ~200, cut at
        #   punctuation/space)
        # - Embed all sub-chunks in batches (<= max_batch_size)
        # - For each original text, length-weighted mean of L2-normalized
        #   sub-embeddings, then L2-normalize the result, returning exactly one vector
        #   per input.
        #
        # Strategy when pooling=False:
        # - Truncate text to max_chars_per_item and log error if truncated
        # - No chunking or pooling
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # TODO: Consider another place for these helpers
        def _l2norm(vec: list[float]) -> list[float]:
            """Perform L2-normalization on a vector."""
            import math

            n = math.sqrt(sum(x * x for x in vec))
            if n == 0.0:
                return vec
            return [x / n for x in vec]

        def _pool(embeds: list[list[float]], weights: list[int]) -> list[float]:
            """Pool embeddings by length-weighted mean.

            Algo: L2-normalize each, length-weighted mean, then final L2-normalize
            """
            if not embeds:
                raise ValueError("No embeddings to pool")
            total_w = float(sum(max(1, w) for w in weights))
            normed = [_l2norm(e) for e in embeds]
            dim = len(normed[0])
            acc = [0.0] * dim
            for e, w in zip(normed, weights, strict=True):
                ww = float(max(1, w)) / total_w
                for i in range(dim):
                    acc[i] += ww * e[i]
            return _l2norm(acc)

        def _preferred_cut(
            text: str, start: int, target: int, max_len: int, window: int = 200
        ) -> int:
            """Find a preferred cut point in text.

            Algo: look for sentence-ish punctuation, double space, or space
            within +/- `window` of `target`, but do not exceed `max_len`.
            """
            # Try to find a boundary near `target` within +/- window
            end_limit = min(len(text), start + max_len)
            # candidate region
            left = max(start + 1, target - window)
            right = min(end_limit, target + window)
            best = -1
            # preference order: sentence-ish punctuation, double space, space
            prefs = {".", "?", "!", ";", ":", ","}
            for i in range(right, left - 1, -1):
                ch = text[i - 1]
                if ch in prefs:
                    best = i
                    break
                # double space (useful when embedding an extracted table)
                if i >= 2 and text[i - 2 : i] == "  ":
                    best = i
                    break
                if ch == " ":
                    if best == -1:
                        best = i
            if best != -1:
                return best
            return min(end_limit, start + max_len)

        def _split_text(s: str) -> tuple[list[str], list[int]]:
            """Split text into sub-chunks.

            This helper is used to comply with Snowflake's 2000 max chars per request
            (which corresponds to ~500 tokens within the 512 token model limit).

            Algo: greedy chunking with target ~1600 chars, overlap ~200, cut at
            punctuation/space. Merge tiny tail if needed.
            """
            # Returns (sub_chunks, weights)
            if len(s) <= (self.max_chars_per_item or 2000):
                return [s], [len(s)]
            subs: list[str] = []
            weights: list[int] = []
            i = 0
            max_len = int(self.max_chars_per_item or 2000)
            target = 1600  # aim for headroom within 2000 char limit
            overlap = 200
            while i < len(s):
                chunk_target_end = i + target
                cut = _preferred_cut(s, i, chunk_target_end, max_len)
                sub = s[i:cut]
                # guard: empty slice (shouldn't happen)
                if not sub:
                    break
                subs.append(sub)
                weights.append(len(sub))
                if cut >= len(s):
                    break
                # advance with overlap
                i = max(i + 1, cut - overlap)
            # merge tiny tail if needed
            if len(subs) >= 2 and len(subs[-1]) < 500:
                subs[-2] = subs[-2] + subs[-1]
                weights[-2] = len(subs[-2])
                subs.pop()
                weights.pop()
            return subs, weights

        def _truncate_text(s: str) -> str:
            """Truncate text to max_chars_per_item and log if truncated."""
            max_len = int(self.max_chars_per_item or 2000)
            if len(s) <= max_len:
                return s
            # Log error when truncating
            logger.error(
                "Text truncated from %d to %d chars (pooling=False). "
                "Content may be lost. First 100 chars: %s",
                len(s),
                max_len,
                s[:100],
            )
            return s[:max_len]

        # Handle pooling=False: truncate texts instead of chunking
        if not pooling:
            truncated_texts = [_truncate_text(t) for t in texts]
            flat_texts = truncated_texts
            # Each text maps to exactly one embedding (no pooling)
            groups = [(i, 1, [len(t)]) for i, t in enumerate(truncated_texts)]
        else:
            # 1) Expand each input into sub-chunks, record mapping
            flat_texts = []
            groups = []  # (start_idx, count, weights)
            for t in texts:
                subs, wts = _split_text(t)
                start_idx = len(flat_texts)
                flat_texts.extend(subs)
                groups.append((start_idx, len(subs), wts))

        # 2) Embed sub-chunks in batches (respect max_batch_size)
        sub_embeddings: list[list[float]] = []
        if not flat_texts:
            return []
        batch_size = int(self.max_batch_size or 1280)
        async with httpx.AsyncClient(timeout=45.0) as client:
            for i in range(0, len(flat_texts), batch_size):
                batch = flat_texts[i : i + batch_size]
                payload = {"text": batch, "model": self._model}
                resp = await client.post(self._base_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                for item in data.get("data", []):
                    embedding = item.get("embedding")
                    if embedding is None:
                        continue
                    # normalize API shape variations (list of floats)
                    if (
                        isinstance(embedding, list)
                        and embedding
                        and isinstance(embedding[0], list)
                    ):
                        # flatten nested lists if returned that way
                        flat = []
                        for sub in embedding:
                            if isinstance(sub, list):
                                flat.extend(sub)
                            else:
                                flat.append(sub)
                        embedding = flat
                    sub_embeddings.append([float(v) for v in embedding])
        if len(sub_embeddings) != len(flat_texts):
            raise ValueError("Mismatch between requested and returned sub-embeddings")

        # 3) Pool back to exactly one vector per input text
        result: list[list[float]] = []
        for start_idx, count, wts in groups:
            if count == 1:
                # L2-normalize single vector for consistency
                result.append(_l2norm(sub_embeddings[start_idx]))
            else:
                embeds = sub_embeddings[start_idx : start_idx + count]
                result.append(_pool(embeds, wts))
        return result

    def count_tokens(self, text: str) -> int | None:
        if self._encoder is None:
            return None
        return len(self._encoder.encode(text))


def _chunk_texts(texts: list[str], provider: _BaseEmbeddingProvider) -> list[list[str]]:
    batches: list[list[str]] = []
    batch: list[str] = []
    current_batch_tokens = 0

    for text in texts:
        candidate_tokens = provider.count_tokens(text)
        token_total_with_candidate = current_batch_tokens + (candidate_tokens or 0)
        batch_size_with_candidate = len(batch) + 1

        exceeds_token_limit = (
            provider.max_batch_tokens is not None
            and candidate_tokens is not None
            and token_total_with_candidate >= provider.max_batch_tokens
        )
        exceeds_size_limit = (
            provider.max_batch_size is not None
            and batch_size_with_candidate > provider.max_batch_size
        )

        if batch and (exceeds_token_limit or exceeds_size_limit):
            batches.append(batch)
            batch = []
            current_batch_tokens = 0

        batch.append(text)
        if candidate_tokens is not None:
            current_batch_tokens += candidate_tokens

    if batch:
        batches.append(batch)

    return batches


def _create_embedding_provider() -> _BaseEmbeddingProvider:
    if OPENBB_EMBEDDING_MODEL_PROVIDER in {"openai", "azure"}:
        return _OpenAIEmbeddingProvider(
            provider=OPENBB_EMBEDDING_MODEL_PROVIDER,
            api_key=OPENBB_EMBEDDING_API_KEY,
            base_url=OPENBB_EMBEDDING_BASE_URL,
            model=OPENBB_EMBEDDING_MODEL,
        )
    if OPENBB_EMBEDDING_MODEL_PROVIDER == "snowflake":
        from openbb_ada.utils.auth import get_snowflake_host

        # Check if API key is "snowflake" and replace with OAuth token
        # Also dynamically construct base URL with correct Snowflake host
        _api_key = OPENBB_EMBEDDING_API_KEY
        if _api_key == "snowflake":
            from openbb_ada.utils.auth import get_login_token

            _api_key = get_login_token()

        # Dynamically construct base URL with correct Snowflake host
        snowflake_host = get_snowflake_host()

        _base_url = f"https://{snowflake_host}/api/v2/cortex/inference:embed"
        return _SnowflakeEmbeddingProvider(
            api_key=_api_key,
            base_url=_base_url,
            model=OPENBB_EMBEDDING_MODEL,
        )
    raise ValueError("OPENBB_EMBEDDING_MODEL_PROVIDER is not set")


class VectorDbDocument(BaseModel):
    page_content: str
    metadata: dict = Field(default_factory=dict)


class VectorDb:
    def __init__(
        self,
        load_from_path: str | Path | None = None,
    ):
        """Use FAISS as a vector database for similarity search.

        Use this class to create a FAISS index, add documents to the index,
        generate embeddings using embedding models, and perform similarity
        searches against the stored vectors.

        You can learn more about FAISS from here:
        https://github.com/facebookresearch/faiss/

        Parameters
        ----------
        model : str, optional
            The model to use for generating embeddings. Default is
            "text-embedding-3-small".
        load_from_path : str or Path, optional
            Path to load an existing FAISS index from. If not provided, a new
            index is created once embeddings are available.

        """

        self._docs: list[VectorDbDocument] = []
        self._provider = _create_embedding_provider()
        self._dimensions: int | None = None
        self._index: faiss.IndexFlatL2 | None = None

        if load_from_path:
            self.load_from_disk(load_from_path)

    @property
    def model(self) -> str:
        return self._provider._model

    async def _get_embeddings(
        self, texts: list[str], *, pooling: bool = True
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0))

        provider = self._provider
        normalized_texts = []
        for original_text in texts:
            text = original_text.replace("\n", " ")
            # Providers can reject a single string if it is too long, regardless of
            # batching, so we guard per-item limits before building the batches.
            # Skip this check if provider can split items (handled internally)
            if (
                provider.max_chars_per_item is not None
                and len(text) > provider.max_chars_per_item
                and not provider.can_split_items
            ):
                raise ValueError(
                    "Text length exceeds provider limit of "
                    f"{provider.max_chars_per_item} characters"
                )

            candidate_tokens = provider.count_tokens(text)
            if (
                # This second guard checks the token estimate for that same string.
                # Even though batching chunks the workload, an individual item can still
                # be too large to send.
                # Example: OpenAI's embedding endpoint will refuse a single prompt whose
                # token count exceeds its limit, regardless of batching.
                provider.max_batch_tokens is not None
                and candidate_tokens is not None
                and candidate_tokens > provider.max_batch_tokens
            ):
                raise ValueError(
                    f"Text exceeds provider token limit of {provider.max_batch_tokens}"
                )

            normalized_texts.append(text)

        batches = _chunk_texts(normalized_texts, provider)
        tasks = [provider.embed(batch, pooling=pooling) for batch in batches]
        responses = await asyncio.gather(*tasks)
        embeddings = [embedding for response in responses for embedding in response]
        return np.array(embeddings)

    async def add(
        self, documents: list[VectorDbDocument], *, pooling: bool = True
    ) -> int:
        """Add a list of documents to the index.

        These indexed documents can be searched using the `search` method.

        Parameters
        ----------
        documents : list[Document]
            A list of Document objects to be added to the FAISS index.
        pooling : bool, optional
            If True (default), chunk long texts and pool embeddings.
            If False, truncate long texts (with error logging) - use for metadata.

        Returns
        -------
        int
            The number of documents added to the FAISS index.
        """
        if not documents:
            raise ValueError("No documents to add to the index")

        logger.debug(
            "VectorDb.add: adding %d documents (pooling=%s)",
            len(documents),
            pooling,
        )
        texts = [document.page_content for document in documents]
        embeddings = await self._get_embeddings(texts, pooling=pooling)
        if embeddings.size == 0:
            logger.debug("VectorDb.add: no embeddings generated")
            return 0

        if self._index is None:
            self._dimensions = embeddings.shape[1]
            self._index = faiss.IndexFlatL2(self._dimensions)

        self._index.add(embeddings)
        self._docs.extend(documents)
        logger.debug(
            "VectorDb.add: index now contains %d documents",
            len(self._docs),
        )
        return len(documents)

    async def search(self, query: str, k: int = 5) -> list[VectorDbDocument]:
        """Search the index for the `k` most similar documents to the query.

        Take a query string, generate its embedding using the specified
        model, and search the index for the top `k` most similar documents.

        Parameters
        ----------
        query : str
            The query string to search for in the index.
        k : int, optional
            The number of top similar documents to return. Default is 4.

        Returns
        -------
        list[Document]
            A list of the top `k` most similar Document objects from the FAISS index.

        """
        results = await self.search_with_scores(query, k)
        return [doc for doc, _ in results]

    async def search_with_scores(
        self, query: str, k: int = 5
    ) -> list[tuple[VectorDbDocument, float]]:
        """Search the index and return documents with their similarity scores.

        Take a query string, generate its embedding using the specified
        model, and search the index for the top `k` most similar documents
        along with their L2 distance scores.

        Parameters
        ----------
        query : str
            The query string to search for in the index.
        k : int, optional
            The number of top similar documents to return. Default is 5.

        Returns
        -------
        list[tuple[VectorDbDocument, float]]
            A list of tuples containing (document, distance_score).
            Lower distance scores indicate higher similarity.

        """
        if not self._docs:
            raise ValueError(
                "No documents to search. Have you tried adding them first?"
            )

        if self._index is None:
            raise ValueError("Vector index is not initialized")

        logger.debug(
            "VectorDb.search_with_scores: query_length=%d, k=%d, docs=%d",
            len(query),
            k,
            len(self._docs),
        )
        query_embedding = await self._get_embeddings(texts=[query])
        distances, indices = self._index.search(query_embedding, k)
        # make sure indices are >= 0, since -1 is returned for duplicate indices
        # when k > number of documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:
                results.append((self._docs[idx], float(distances[0][i])))
        logger.debug(
            "VectorDb.search_with_scores: returning %d result(s)",
            len(results),
        )
        return results

    def save_to_buffer(self, buffer: io.BytesIO) -> io.BytesIO:
        if self._index is None:
            raise ValueError("Vector index is not initialized")
        with zipfile.ZipFile(buffer, "w") as zip_file:
            # We need to allow FAISS to write to a file: a buffer doesn't work
            # due to the underlying C++ code.
            with tempfile.NamedTemporaryFile() as temp_file:
                faiss.write_index(self._index, temp_file.name)
                temp_file.seek(0)
                zip_file.writestr("vector_index.index", temp_file.read())
            zip_file.writestr(
                "documents.json", json.dumps(jsonable_encoder(self._docs))
            )
        buffer.seek(0)
        return buffer

    def save_to_disk(self, path: Path | str) -> None:
        path = Path(path)
        if not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise IOError(f"The path {path} must be a directory.")

        if self._index is None:
            raise ValueError("Vector index is not initialized")

        # Vector index
        faiss.write_index(self._index, str(path / "vector_index.index"))
        # Document index
        with open(path / "documents.json", "w") as f:
            # We're using a FastAPI convenience function to make the pydantic
            # Document objects in the document index json-serializable
            json.dump(jsonable_encoder(self._docs), f)

    def load_from_disk(self, path: Path | str) -> None:
        path = Path(path)
        if not path.is_dir():
            raise IOError(f"The path {path} must be a directory.")
        elif len([item for item in path.iterdir()]) == 0:
            raise IOError(f"The path {path} is empty.")
        self._index = faiss.read_index(str(path / "vector_index.index"))
        self._dimensions = self._index.d
        with open(path / "documents.json") as f:
            self._docs = [VectorDbDocument(**doc) for doc in json.load(f)]

    def merge_from(self, other: "VectorDb") -> None:
        if other._index is None:
            raise ValueError("Source vector index must be initialized")

        if self._index is None:
            self._index = faiss.clone_index(other._index)
            self._dimensions = other._dimensions or other._index.d
            self._docs = [VectorDbDocument(**doc.model_dump()) for doc in other._docs]
            return

        self._index.merge_from(other._index)
        self._docs.extend(VectorDbDocument(**doc.model_dump()) for doc in other._docs)

    @property
    def docs(self):
        return self._docs
