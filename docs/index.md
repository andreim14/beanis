---
title: Beanis
hide:
  - navigation
---

# Beanis { .page-title-hidden }

<img class="hero-logo" src="assets/beanis.svg#only-light" alt="beanis">
<img class="hero-logo" src="assets/beanis-dark.svg#only-dark" alt="beanis">

Beanis is an async object-document mapper for Redis, built on Pydantic. You define a model, and Beanis stores each instance as a Redis hash, keeps secondary indexes up to date, and gives you typed objects back.

It also does vector search on the vector sets that ship with Redis 8, and it includes long-term memory for AI agents, usable from Python, from the `beanis` command line, or as an MCP server for Claude Code and other agents. It works on plain Redis: no RedisJSON, RediSearch or Redis Stack.

[Get started](getting-started.md){ .md-button .md-button--primary } [Agent memory and MCP](agents.md){ .md-button } [GitHub](https://github.com/andreim14/beanis){ .md-button }

## Install

```bash
pip install beanis               # the ODM
pip install "beanis[mcp]"        # plus agent memory: CLI, embeddings and MCP server (Python 3.10+)
```

Redis 5+ is enough for documents and indexes; vector search and memory need Redis 8.0+:

```bash
docker run -d -p 6379:6379 redis:8      # or: brew install redis
```

## Quick start

```python
import asyncio

from redis.asyncio import Redis

from beanis import Document, Indexed, init_beanis


class Product(Document):
    name: str
    price: Indexed(float)            # sorted-set index: range queries
    category: Indexed(str)           # set index: exact match
    tags: list[str] = []
    stock: int = 0

    class Settings:
        key_prefix = "Product"


async def main():
    redis = Redis(decode_responses=True)
    await init_beanis(database=redis, document_models=[Product])

    bar = Product(name="Dark chocolate", price=5.95, category="chocolate", stock=100)
    await bar.insert()

    same = await Product.get(bar.id)
    cheap = await Product.find(price__lt=10)
    chocolate = await Product.find(category="chocolate", price__gte=5)

    await bar.update(price=6.50)
    await bar.increment_field("stock", -1)
    await bar.delete_self()


asyncio.run(main())
```

`decode_responses=True` is required. Each document is the hash `Product:<id>`; nested models, lists, dicts, `datetime`, `Decimal`, `UUID` and enums are encoded for you and decoded on read.

## What it covers

<div class="grid cards" markdown>

-   **Documents and queries**

    Pydantic models as Redis hashes, with `find()` on indexed fields, TTL, batches and field-level updates.

    [Tutorial](tutorial/README.md)

-   **Vector search**

    Embeddings in Redis 8 vector sets, filtered by the document's fields, in about a millisecond.

    [Vector search](vector-search.md)

-   **Agent memory**

    Remember, recall and forget, scoped per project, from Python, the CLI or MCP.

    [Agent memory and MCP](agents.md)

-   **Geo search**

    `GeoPoint` fields with radius queries on Redis geo indexes.

    [Geo search](tutorial/geo-spatial.md)

</div>

## When to use it

Use Beanis when you want typed models on Redis without running Redis modules: caches with structure, session and user state, job metadata, small catalogues, retrieval for RAG, and agent memory.

Use something else when you need complex queries across many fields, full-text search or aggregations: a document database, or Redis's query engine with [Redis OM](https://github.com/redis/redis-om-python) or [RedisVL](https://github.com/redis/redis-vl-python).

Beanis started as a fork of [Beanie](https://github.com/BeanieODM/beanie), the MongoDB ODM by Roman Right and contributors, rewritten for Redis.
