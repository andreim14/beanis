"""
Regression tests that need a real Redis server (8.0+ for vector sets).

Run with a server on REDIS_PORT (default 6379); they are skipped when none is reachable.
fakeredis does not implement every command these paths use (vector sets in particular).
"""
from typing import List, Optional

import pytest
from typing_extensions import Annotated

from beanis import Document, IndexedField, VectorField, init_beanis


class Note(Document):
    text: str
    topic: Annotated[str, IndexedField()]
    priority: Annotated[int, IndexedField()] = 0
    embedding: Annotated[List[float], VectorField(dimensions=3)]

    class Settings:
        key_prefix = "Note"


async def _init(client):
    await init_beanis(database=client, document_models=[Note])


@pytest.mark.asyncio
async def test_vector_search_returns_inserted_documents(redis_client_real):
    await _init(redis_client_real)
    await Note(id="a", text="cats", topic="pets", embedding=[1.0, 0.0, 0.0]).insert()
    await Note(id="b", text="stocks", topic="money", embedding=[0.0, 1.0, 0.0]).insert()

    hits = await Note.find_by_vector_similarity([0.9, 0.1, 0.0], k=2)

    assert [doc_id for doc_id, _ in hits][0] == "a"
    assert all(isinstance(score, float) for _, score in hits)


@pytest.mark.asyncio
async def test_document_with_vector_round_trips(redis_client_real):
    await _init(redis_client_real)
    await Note(id="a", text="cats", topic="pets", embedding=[0.25, 0.5, 0.75]).insert()

    loaded = await Note.get("a")

    assert loaded is not None
    assert loaded.embedding == pytest.approx([0.25, 0.5, 0.75])


@pytest.mark.asyncio
async def test_insert_many_updates_secondary_indexes(redis_client_real):
    await _init(redis_client_real)
    await Note.insert_many(
        [
            Note(id="a", text="x", topic="pets", embedding=[1.0, 0.0, 0.0]),
            Note(id="b", text="y", topic="pets", embedding=[0.0, 1.0, 0.0]),
        ]
    )

    assert {n.id for n in await Note.find(topic="pets")} == {"a", "b"}


@pytest.mark.asyncio
async def test_update_moves_secondary_index(redis_client_real):
    await _init(redis_client_real)
    note = Note(id="a", text="x", topic="pets", embedding=[1.0, 0.0, 0.0])
    await note.insert()

    await note.update(topic="money")

    assert await Note.find(topic="pets") == []
    assert [n.id for n in await Note.find(topic="money")] == ["a"]


@pytest.mark.asyncio
async def test_exclusive_range_filters(redis_client_real):
    await _init(redis_client_real)
    for i, p in enumerate([1, 2, 3]):
        await Note(id=f"n{i}", text="x", topic="t", priority=p, embedding=[1.0, 0.0, 0.0]).insert()

    assert {n.priority for n in await Note.find(priority__gt=1)} == {2, 3}
    assert {n.priority for n in await Note.find(priority__lt=3)} == {1, 2}


@pytest.mark.asyncio
async def test_delete_many_cleans_indexes(redis_client_real):
    await _init(redis_client_real)
    await Note(id="a", text="x", topic="pets", embedding=[1.0, 0.0, 0.0]).insert()
    await Note(id="b", text="y", topic="pets", embedding=[0.0, 1.0, 0.0]).insert()

    await Note.delete_many(["a", "b"])

    assert await redis_client_real.smembers("idx:Note:topic:pets") == set()
    assert await Note.find_by_vector_similarity([1.0, 0.0, 0.0], k=2) == []
