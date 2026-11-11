"""
Long-term memory for agents, on plain Redis 8.

    memory = Memory(redis_client, embed=my_embedding_function)
    await memory.remember("Prefers answers in Italian", user_id="u42")
    for item, score in await memory.recall("which language?", user_id="u42"):
        print(round(score, 3), item.text)

Each memory is a Beanis document; its embedding lives in a Redis vector set,
so recall is one VSIM call filtered by user, session, kind and namespace.
"""

import asyncio
import inspect
import json
import os
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pydantic import Field
from typing_extensions import Annotated

from beanis.odm.documents import Document
from beanis.odm.indexes import IndexedField, VectorField
from beanis.odm.utils.init import init_beanis

Vector = Sequence[float]
EmbedFn = Callable[[str], Union[Vector, Awaitable[Vector]]]


class MemoryRecord(Document):
    text: str
    namespace: Annotated[str, IndexedField()] = "default"
    user_id: Annotated[Optional[str], IndexedField()] = None
    session_id: Annotated[Optional[str], IndexedField()] = None
    kind: Annotated[str, IndexedField()] = "note"
    created_at: float = Field(default_factory=time.time)
    # dimensions=None: fixed by the embedding model on first write
    embedding: Annotated[List[float], VectorField(dimensions=None)]

    class Settings:
        key_prefix = "beanis:memory"


def _eq(field: str, value: str) -> str:
    return f".{field} == {json.dumps(value)}"


class Memory:
    """
    Agent memory: remember facts, recall the relevant ones, forget on request.

    :param redis_client: redis.asyncio client (Redis 8+, decode_responses=True)
    :param embed: text -> vector, sync or async (any embedding model)
    :param namespace: separates memories of different agents or apps
    :param dedupe_threshold: a new memory this similar (0-1) to an existing one
        for the same user/session/kind replaces it instead of adding a copy;
        set to None to always add
    """

    def __init__(
        self,
        redis_client: Any,
        embed: EmbedFn,
        namespace: str = "default",
        dedupe_threshold: Optional[float] = 0.95,
    ):
        self.redis = redis_client
        self._embed = embed
        self.namespace = namespace
        self.dedupe_threshold = dedupe_threshold
        self._ready = False

    async def setup(self) -> "Memory":
        if not self._ready:
            await init_beanis(
                database=self.redis, document_models=[MemoryRecord]
            )
            self._ready = True
        return self

    async def embed(self, text: str) -> List[float]:
        vector = self._embed(text)
        if inspect.isawaitable(vector):
            vector = await vector
        return [float(v) for v in vector]

    def _filter(
        self,
        user_id: Optional[str],
        session_id: Optional[str],
        kind: Optional[str],
    ) -> str:
        parts = [_eq("namespace", self.namespace)]
        if user_id is not None:
            parts.append(_eq("user_id", user_id))
        if session_id is not None:
            parts.append(_eq("session_id", session_id))
        if kind is not None:
            parts.append(_eq("kind", kind))
        return " and ".join(parts)

    async def remember(
        self,
        text: str,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        kind: str = "note",
        ttl: Optional[int] = None,
    ) -> MemoryRecord:
        """Store a memory. Returns the stored (or merged) record."""
        await self.setup()
        vector = await self.embed(text)
        if self.dedupe_threshold is not None:
            hits = await MemoryRecord.vector_search(
                vector,
                k=1,
                filter=self._filter(user_id, session_id, kind),
                min_score=self.dedupe_threshold,
            )
            if hits:
                existing, _ = hits[0]
                existing.text = text
                existing.embedding = vector
                existing.created_at = time.time()
                return await existing.insert(ttl=ttl)
        record = MemoryRecord(
            text=text,
            namespace=self.namespace,
            user_id=user_id,
            session_id=session_id,
            kind=kind,
            embedding=vector,
        )
        return await record.insert(ttl=ttl)

    async def recall(
        self,
        query: str,
        *,
        k: int = 5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        kind: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Most relevant memories for ``query``, best first, with similarity 0-1."""
        await self.setup()
        return await MemoryRecord.vector_search(
            await self.embed(query),
            k=k,
            filter=self._filter(user_id, session_id, kind),
            min_score=min_score,
        )

    async def forget(self, memory_id: str) -> bool:
        await self.setup()
        record = await MemoryRecord.get(memory_id)
        if record is None or record.namespace != self.namespace:
            return False
        await record.delete_self()
        return True

    async def recent(
        self,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 20,
    ) -> List[MemoryRecord]:
        """Newest memories first, optionally scoped to a user/session/kind."""
        await self.setup()
        filters = {"namespace": self.namespace}
        if user_id is not None:
            filters["user_id"] = user_id
        if session_id is not None:
            filters["session_id"] = session_id
        if kind is not None:
            filters["kind"] = kind
        records = await MemoryRecord.find(**filters)
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]


DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def fastembed_embedder(model_name: str = DEFAULT_EMBED_MODEL) -> EmbedFn:
    """Local CPU embeddings via fastembed (pip install fastembed); no API key."""
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise ImportError(
            'Local embeddings need fastembed: pip install "beanis[mcp]"'
        ) from e

    model = TextEmbedding(model_name)

    async def embed(text: str) -> List[float]:
        vec = await asyncio.to_thread(lambda: next(iter(model.embed([text]))))
        return vec.tolist()

    return embed


def memory_from_env(namespace: Optional[str] = None) -> Memory:
    """
    Memory configured from the environment, as used by the CLI and MCP server.

    REDIS_URL (redis://localhost:6379/0), BEANIS_MEMORY_NAMESPACE (default),
    BEANIS_EMBED_MODEL (BAAI/bge-small-en-v1.5). All memories share one
    vector set, so every writer must use the same embedding model.
    """
    from redis.asyncio import Redis

    client = Redis.from_url(
        os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
    )
    return Memory(
        client,
        embed=fastembed_embedder(
            os.environ.get("BEANIS_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        ),
        namespace=namespace
        or os.environ.get("BEANIS_MEMORY_NAMESPACE", "default"),
    )
