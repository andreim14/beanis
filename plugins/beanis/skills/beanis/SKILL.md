---
name: beanis
description: Write correct code with Beanis, the async Pydantic ODM for Redis (documents, indexes, find, vector search on Redis 8 vector sets, agent Memory, beanis CLI and MCP server). Use when code imports beanis, defines a beanis Document, or the user asks to store, query or vector-search data in Redis with Beanis, or to give an agent long-term memory with Beanis.
---

# Beanis

Beanis stores Pydantic models as Redis hashes and keeps secondary indexes. It is NOT Beanie (MongoDB): there is no query builder, no `to_list()`, `find_one()`, `.sort()`, `.limit()` and no `Model.field > x` expressions. Use only the API below.

## Setup

```python
from redis.asyncio import Redis
from beanis import Document, Indexed, IndexedField, VectorField, GeoPoint, init_beanis

redis = Redis(decode_responses=True)          # decode_responses=True is required
await init_beanis(database=redis, document_models=[Product])
```

Requirements: Python 3.9+, Redis 5+; Redis 8.0+ for `VectorField` and `Memory` (vector sets). `init_beanis` raises if a model has vector fields and the server lacks `VADD`. Run `beanis doctor` to check a server.

## Models and indexes

```python
from typing import List, Optional
from typing_extensions import Annotated

class Product(Document):
    name: str
    price: Indexed(float)                                   # number -> sorted set (ranges)
    category: Annotated[str, IndexedField()]                # scalar -> set (exact match)
    location: Indexed(GeoPoint)                             # geo index (not Optional[...])
    embedding: Annotated[List[float], VectorField(dimensions=384)]  # Redis 8 vector set

    class Settings:
        key_prefix = "Product"                              # hash key: Product:<id>
```

- `Indexed` is a function: `Indexed(float)`, never `Indexed[float]`.
- Define models at module level (decoding looks the class up by name).
- Nested BaseModels, lists, dicts, datetime, Decimal, UUID, enums are encoded automatically; `register_type(cls, encoder=, decoder=)` for others.

## CRUD

```python
doc = await Product(name="Tea", price=3.0, category="drinks", embedding=vec).insert()   # insert(ttl=seconds) optional
await Product.get(doc.id)                 # None if missing
await Product.get_many(ids); await Product.exists(id); await Product.count()
await Product.insert_many(docs)
await doc.save()                          # upsert; moves index entries if indexed values changed
await doc.update(price=4.0)               # partial update, indexes kept in sync
await doc.get_field("price"); await doc.set_field("price", 5.0); await doc.increment_field("stock", -1)
await doc.set_ttl(60); await doc.get_ttl(); await doc.persist()
await doc.delete_self(); await Product.delete(id); await Product.delete_many(ids); await Product.delete_all()
```

## Queries: keyword filters on indexed fields only

```python
await Product.find(category="drinks")                 # exact
await Product.find(price__gte=10, price__lte=50)      # __gte __lte __gt __lt
await Product.find(price=12.0)                        # exact number
await Product.find(category="drinks", price__lt=5)    # ANDed
await Product.all(skip=0, limit=20)                   # by insertion time
```

A filter on a non-indexed field raises `ValueError`. Sort or slice the returned list in Python. Geo: `IndexManager.find_by_geo_radius(...)` from `beanis.odm.indexes` (see docs/tutorial/geo-spatial.md).

## Vector search

```python
hits = await Product.vector_search(query_vec, k=5, filter='.category == "drinks"', min_score=0.0)
for doc, score in hits: ...               # best first; score = cosine similarity in [0, 1]
```

`filter` uses vector-set filter syntax over the document's short scalar fields: `.field == "x"`, `.price > 10`, `and`, `or`, `not`. HNSW, cosine only, int8 quantization by default (`VectorField(quantization="NOQUANT")` for float32). `Product.find_by_vector_similarity(...)` returns `(id, score)` pairs without loading documents.

## Agent memory

```python
from beanis import Memory
from beanis.memory import fastembed_embedder, memory_from_env

memory = Memory(redis, embed=fastembed_embedder(), namespace="my-agent")  # or: memory_from_env()
rec = await memory.remember("User prefers metric units", user_id="u1", kind="preference", ttl=None)
hits = await memory.recall("which units?", user_id="u1", k=5, min_score=0.0)   # [(MemoryRecord, score)]
await memory.recent(user_id="u1", limit=10)
await memory.forget(rec.id)
```

Near-duplicates (similarity >= 0.95 in the same scope) replace the old memory. All memories share one vector set, so every writer must use the same embedding model.

## When the memory MCP tools are available

If tools named `remember`, `recall`, `forget` and `recent` from the beanis memory server are available:

- Call `recall` before answering questions that may depend on earlier conversations, project decisions or user preferences.
- Call `remember` when the user states a durable fact, preference or decision, or when you settle something worth knowing next session (a convention, a gotcha, where something lives). Write one self-contained sentence; set `kind` to `preference`, `decision`, `fact` or `note`.
- Do not store secrets, credentials or personal data the user has not asked you to keep.
- `forget` outdated memories by id when the user corrects them.

## CLI (JSON on stdout)

```bash
beanis doctor
beanis memory remember "text" [--user U] [--session S] [--kind K] [--ttl N]
beanis memory recall "query" [-k 5] [--user U] [--kind K] [--min-score 0.3]
beanis memory recent [--limit 10]
beanis memory forget <id>
beanis mcp            # stdio MCP server, same as beanis-mcp
```

Environment: `REDIS_URL`, `BEANIS_MEMORY_NAMESPACE`, `BEANIS_EMBED_MODEL`.
