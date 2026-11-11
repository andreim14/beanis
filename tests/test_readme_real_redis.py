"""The README examples, run against a real Redis 8 server (REDIS_PORT)."""

import hashlib
import math
import re
from typing import List

import pytest
from pydantic import BaseModel
from typing_extensions import Annotated

from beanis import (
    Document,
    Indexed,
    IndexedField,
    Insert,
    Memory,
    VectorField,
    before_event,
    init_beanis,
)


def embed(text: str) -> List[float]:
    vec = [0.0] * 32
    for word in re.findall(r"[a-z]+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 32] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class Category(BaseModel):
    name: str
    description: str


class ReadmeProduct(Document):
    name: str
    price: Indexed(float)
    category: Indexed(str)
    tags: List[str] = []
    stock: int = 0

    @before_event(Insert)
    def check_price(self):
        if self.price < 0:
            raise ValueError("price cannot be negative")

    class Settings:
        key_prefix = "Product"


class ReadmeChunk(Document):
    text: str
    source: Annotated[str, IndexedField()]
    embedding: Annotated[List[float], VectorField(dimensions=32)]


@pytest.mark.asyncio
async def test_quick_start_and_queries(redis_client_real):
    await init_beanis(
        database=redis_client_real, document_models=[ReadmeProduct]
    )
    bar = ReadmeProduct(
        name="Dark chocolate", price=5.95, category="chocolate", stock=100
    )
    await bar.insert()
    await ReadmeProduct(
        name="Truffle", price=12.0, category="chocolate"
    ).insert()
    await ReadmeProduct(name="Tea", price=3.0, category="drinks").insert()

    assert await redis_client_real.exists(f"Product:{bar.id}")
    assert (await ReadmeProduct.get(bar.id)).name == "Dark chocolate"
    assert {p.name for p in await ReadmeProduct.find(price__lt=10)} == {
        "Dark chocolate",
        "Tea",
    }
    assert {
        p.name
        for p in await ReadmeProduct.find(category="chocolate", price__gte=5)
    } == {
        "Dark chocolate",
        "Truffle",
    }
    assert [p.name for p in await ReadmeProduct.find(price=12.0)] == [
        "Truffle"
    ]
    assert len(await ReadmeProduct.all(skip=0, limit=20)) == 3
    assert await ReadmeProduct.count() == 3
    with pytest.raises(ValueError):
        await ReadmeProduct.find(stock=1)
    with pytest.raises(ValueError):
        await ReadmeProduct(name="Bad", price=-1, category="x").insert()

    await bar.update(price=6.50)
    assert await bar.increment_field("stock", -1) == 99
    await bar.insert(ttl=3600)
    assert 0 < await bar.get_ttl() <= 3600
    assert await bar.get_field("price") == 6.5
    await bar.delete_self()
    assert await ReadmeProduct.get(bar.id) is None


@pytest.mark.asyncio
async def test_vector_search(redis_client_real):
    await init_beanis(
        database=redis_client_real, document_models=[ReadmeChunk]
    )
    for text, source in [
        ("Refunds take 5 days", "faq"),
        ("Refunds are free", "blog"),
    ]:
        await ReadmeChunk(
            text=text, source=source, embedding=embed(text)
        ).insert()

    hits = await ReadmeChunk.vector_search(
        embed("how long do refunds take"), k=3, filter='.source == "faq"'
    )

    assert [c.text for c, _ in hits] == ["Refunds take 5 days"]
    assert 0 < hits[0][1] <= 1


@pytest.mark.asyncio
async def test_memory(redis_client_real):
    memory = Memory(redis_client_real, embed=embed, namespace="support-bot")
    await memory.remember(
        "Prefers answers in Italian", user_id="u42", kind="preference"
    )

    hits = await memory.recall(
        "which language should I answer in Italian", user_id="u42"
    )
    record = hits[0][0]
    assert record.text == "Prefers answers in Italian"
    assert [r.id for r in await memory.recent(user_id="u42")] == [record.id]
    assert await memory.forget(record.id)
