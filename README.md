<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/andreim14/beanis/main/assets/logo/beanis-dark.svg">
    <img src="https://raw.githubusercontent.com/andreim14/beanis/main/assets/logo/beanis.svg" alt="beanis" width="420">
  </picture>
</p>

<div align="center">
<a href="https://pypi.python.org/pypi/beanis"><img src="https://img.shields.io/pypi/v/beanis?color=blue" alt="PyPI version"></a>
<a href="https://pypi.python.org/pypi/beanis"><img src="https://img.shields.io/pypi/pyversions/beanis?color=blue" alt="Python versions"></a>
<a href="https://github.com/andreim14/beanis/blob/main/LICENSE"><img src="https://img.shields.io/github/license/andreim14/beanis?color=blue" alt="License"></a>
</div>

Beanis is an async object-document mapper for Redis, built on Pydantic. You define a model, and Beanis stores each instance as a Redis hash, keeps secondary indexes up to date, and gives you typed objects back.

It also does vector search on the vector sets that ship with Redis 8, and it includes long-term memory for AI agents, usable from Python, from the `beanis` command line, or as an MCP server for Claude Code and other agents.

It works on plain Redis. No RedisJSON, RediSearch or Redis Stack.

[Documentation](https://andreim14.github.io/beanis/) · [PyPI](https://pypi.org/project/beanis/) · [Changelog](docs/changelog.md)

## Install

```bash
pip install beanis               # the ODM
pip install "beanis[mcp]"        # plus agent memory: CLI embeddings and MCP server (Python 3.10+)
```

You need a Redis server. Redis 5+ is enough for documents and indexes; vector search and memory need Redis 8.0+:

```bash
docker run -d -p 6379:6379 redis:8      # or: brew install redis
```

## Quick start

```python
import asyncio

from pydantic import BaseModel
from redis.asyncio import Redis

from beanis import Document, Indexed, init_beanis


class Category(BaseModel):
    name: str
    description: str


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

## Queries

`find()` takes keyword filters on indexed fields. Several filters are combined with AND.

```python
await Product.find(category="chocolate")              # exact match
await Product.find(price__gte=10, price__lte=50)      # range: __gte, __lte, __gt, __lt
await Product.find(price=12.0)                        # exact match on a number
await Product.all(skip=0, limit=20)                   # every document, by insertion time
await Product.count()
```

Filtering on a field that is not indexed raises `ValueError`. Sorting by other fields and full-text search are not built in; sort the returned list in Python, or use Redis's query engine directly if you need them.

Declare indexes with `Indexed(type)` or `Annotated[type, IndexedField()]`. Numbers get a sorted set, strings and other scalars get a set, `GeoPoint` gets a geo index (see the [geo tutorial](docs/tutorial/geo-spatial.md)).

## Vector search

Embeddings live in a Redis 8 vector set (`VADD` / `VSIM`), next to the document's short fields so you can filter a search:

```python
from typing import List

from typing_extensions import Annotated

from beanis import Document, IndexedField, VectorField


class Chunk(Document):
    text: str
    source: Annotated[str, IndexedField()]
    embedding: Annotated[List[float], VectorField(dimensions=384)]


await Chunk(text="Refunds take 5 days", source="faq", embedding=embed("Refunds take 5 days")).insert()

for chunk, score in await Chunk.vector_search(embed("how long is a refund?"), k=3, filter='.source == "faq"'):
    print(round(score, 3), chunk.text)        # score: cosine similarity in [0, 1]
```

`embed` is any function from text to a list of floats. Vectors use HNSW with int8 quantization by default (`VectorField(quantization="NOQUANT")` keeps float32). Updating or deleting a document updates its vector. `init_beanis` raises at startup if a model has vector fields and the server has no vector sets (Redis 7, Valkey).

## Memory for agents

`Memory` stores what an agent should keep between conversations and finds it again by meaning. Memories are scoped by namespace, user, session and kind; a new memory that is nearly identical to an old one replaces it instead of piling up.

```python
from beanis import Memory
from beanis.memory import fastembed_embedder

memory = Memory(redis, embed=fastembed_embedder(), namespace="support-bot")

await memory.remember("Prefers answers in Italian", user_id="u42", kind="preference")
for record, score in await memory.recall("which language should I answer in?", user_id="u42"):
    print(round(score, 3), record.text)
await memory.recent(user_id="u42")
await memory.forget(record.id)
```

`fastembed_embedder()` runs `BAAI/bge-small-en-v1.5` locally on the CPU (no API key). Any embedding function works, sync or async; use the same one for every writer of a namespace.

### From the command line

`pip install "beanis[mcp]"` installs a `beanis` command. Every command prints JSON, so shell scripts and coding agents can use it directly.

```bash
beanis doctor                                   # checks Redis, vector sets, extras, stored memory
beanis memory remember "Deploys go out on Tuesdays" --kind decision
beanis memory recall "when do we deploy?" -k 3
beanis memory recent --limit 5
beanis memory forget <id>
beanis context                                  # newest memories as plain text, for session hooks
```

Configuration is by environment: `REDIS_URL` (default `redis://localhost:6379/0`, or pass `--url`), `BEANIS_MEMORY_NAMESPACE` (default `default`; `auto` gives each git project its own namespace), `BEANIS_EMBED_MODEL` (default `BAAI/bge-small-en-v1.5`).

### As an MCP server

`beanis-mcp` (or `beanis mcp`) serves the same memory over MCP stdio with four tools: `remember`, `recall`, `forget`, `recent`.

Claude Code, as a plugin (MCP server plus a skill that teaches Claude the Beanis API):

```bash
claude plugin marketplace add andreim14/beanis
claude plugin install beanis@beanis
```

The plugin gives each project its own namespace and, at the start of every session, puts that project's newest memories into Claude's context, so a new session does not start from zero. Sessions and subagents share the store: what one remembers, the next one knows. It starts everything with `uvx`, so it needs [uv](https://docs.astral.sh/uv/) and a Redis 8 at `REDIS_URL`. Without the plugin:

```bash
claude mcp add beanis-memory --env REDIS_URL=redis://localhost:6379/0 -- beanis-mcp
```

Claude Desktop, Cursor and other MCP clients:

```json
{
  "mcpServers": {
    "beanis-memory": {
      "command": "beanis-mcp",
      "env": { "REDIS_URL": "redis://localhost:6379/0" }
    }
  }
}
```

## More

```python
await product.insert(ttl=3600)                 # expire after an hour
await product.set_ttl(7200); await product.get_ttl(); await product.persist()

price = await product.get_field("price")       # read or write one field
await product.set_field("stock", 200)

await Product.insert_many(products)            # pipelined batches
await Product.get_many(ids)
await Product.delete_many(ids)
```

Event hooks run around inserts, updates and deletes:

```python
from beanis import Document, Insert, before_event


class Product(Document):
    name: str
    price: float

    @before_event(Insert)
    def check_price(self):
        if self.price < 0:
            raise ValueError("price cannot be negative")
```

Custom types are registered once with `register_type(cls, encoder=..., decoder=...)`; NumPy arrays are supported when NumPy is installed. See the [custom encoders tutorial](docs/tutorial/custom-encoders.md).

## When to use it

Use Beanis when you want typed models on Redis without running Redis modules: caches with structure, session and user state, job metadata, small catalogues, retrieval for RAG, and agent memory.

Use something else when you need complex queries across many fields, full-text search or aggregations: a document database, or Redis's query engine with [Redis OM](https://github.com/redis/redis-om-python) or [RedisVL](https://github.com/redis/redis-vl-python).

## Requirements

- Python 3.9+ (3.10+ for the `mcp` extra)
- Redis 5.0+; Redis 8.0+ for `VectorField`, `Memory`, the CLI memory commands and the MCP server
- Pydantic 1.10+ or 2.x

## Development

```bash
pip install -e ".[test,mcp]"
pytest                                        # unit tests, on fakeredis
REDIS_PORT=6379 pytest tests/test_real_redis_regressions.py tests/test_memory_real_redis.py   # needs Redis 8
```

See [AGENTS.md](AGENTS.md) for repository conventions.

## Credits

Beanis started as a fork of [Beanie](https://github.com/BeanieODM/beanie), the MongoDB ODM by Roman Right and contributors, rewritten for Redis.

## License

Apache License 2.0
