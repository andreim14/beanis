# Changelog

Beanis - Redis ODM for Python

## [0.3.1] - 2026-11-11

- `beanis` command line: `doctor` (checks Redis, vector sets, extras and the stored memory's embedding size), `memory remember|recall|forget|recent`, `mcp`. Output is JSON.
- Claude Code plugin in this repository: memory MCP server plus a skill with the Beanis API (`claude plugin marketplace add andreim14/beanis`).
- `beanis.memory.memory_from_env()`, shared by the CLI and MCP server.
- Fix: `find(price=12.0)` on a numeric index returned every document.
- Fix: a connection error at `init_beanis` was reported as missing vector sets.
- README, tutorials and docs use the real query API (no Beanie query builder); `AGENTS.md` and `llms.txt` added.
- New logo.

## [0.3.0] - 2026-11-10

- Vector search on Redis 8 vector sets (`VADD`/`VSIM`): `VectorField`, `Document.vector_search`, filters on scalar fields. No RediSearch needed.
- Index maintenance fixes for `save`, `insert_many`, `update` and `delete_many`.
- `Memory` for agents and the `beanis-mcp` MCP server (`pip install "beanis[mcp]"`).

## [0.1.0] - 2025-01-15

### Major Release - Complete Redis Refactor

Complete refactor from MongoDB (Beanie fork) to Redis-native implementation.

#### Core Features
- ✅ **Redis Hash Storage** - Documents stored as Redis Hashes
- ✅ **Automatic Indexing** - Sorted Sets for numeric fields, Sets for categorical fields
- ✅ **Type Safety** - Full Pydantic v2 validation
- ✅ **Async/Await** - Built on redis.asyncio
- ✅ **TTL Support** - Built-in document expiration
- ✅ **Event Hooks** - Before/after insert, update, delete, save

#### Custom Encoders System
- Author - [Claude Code Assistant](https://github.com/anthropics/claude-code)
- Custom type encoding/decoding registry
- Decorator and function APIs
- Auto-registration for NumPy and PyTorch types
- Type metadata storage for runtime type resolution
- Example: Store NumPy arrays, PyTorch tensors, custom classes

#### Performance Optimizations
- **msgspec** for JSON serialization (2x faster than orjson)
- **Redis pipelines** for batch operations
- **Lazy validation** - Skip validation on reads by default
- **8% overhead** vs vanilla Redis (benchmarked)

#### Documentation
- Complete tutorial system (8 core tutorials)
- Getting started guide
- Custom encoders guide
- Side-by-side code comparisons (vanilla Redis vs Beanis)
- Removed MongoDB-only features documentation

#### API Changes
- `init_beanis()` instead of `init_beanie()`
- Redis client instead of Motor client
- `Indexed(type)` for indexable fields
- Removed: Link, BackLink, migrations, aggregations, views

#### Tests
- 72 passing tests
- Comprehensive document operations tests
- Custom encoder tests
- FastAPI integration tests
- Migration tests (legacy, kept for reference)

[0.1.0]: https://pypi.org/project/beanis/0.1.0

## [0.0.8] - 2024-06-05

### Initial Fork from Beanie

- Forked from [Beanie ODM](https://github.com/BeanieODM/beanie)
- Changed package name to Beanis
- Updated imports and references
- Initial PyPI release

[0.0.8]: https://pypi.org/project/beanis/0.0.8

---

## Credits

Beanis is inspired by [Beanie](https://github.com/BeanieODM/beanie) - the amazing MongoDB ODM by [Roman Right](https://github.com/roman-right).

We took the Beanie philosophy (elegant API, Pydantic models, async/await) and adapted it for Redis, creating a simple yet powerful ODM that works with vanilla Redis.

## Migration from Beanie

If you're migrating from Beanie (MongoDB) to Beanis (Redis), here are the key changes:

### Initialization
```python
# Before (Beanie)
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

client = AsyncIOMotorClient("mongodb://localhost:27017")
await init_beanie(database=client.db_name, document_models=[Product])

# After (Beanis)
from redis.asyncio import Redis
from beanis import init_beanis

client = Redis(decode_responses=True)
await init_beanis(database=client, document_models=[Product])
```

### Document Definition
```python
# Before (Beanie)
from beanie import Document
import pymongo

class Product(Document):
    name: str
    price: Indexed(float, index_type=pymongo.DESCENDING)

    class Settings:
        name = "products"

# After (Beanis)
from beanis import Document, Indexed

class Product(Document):
    name: str
    price: Indexed(float)  # Sorted Set index

    class Settings:
        name = "products"
```

### Features Not Available
- ❌ Relations (Link/BackLink) - Use embedded documents
- ❌ Aggregation pipelines - Use Python for data processing
- ❌ Migrations - Not needed (schema-less)
- ❌ Views - Not applicable to Redis
- ❌ Time Series - Use Redis TimeSeries module or TTL

### New Features
- ✅ TTL support - Document expiration
- ✅ Custom encoders - Store any Python type
- ✅ Atomic operations - increment_field() for counters
- ✅ Performance - 8% overhead vs vanilla Redis
