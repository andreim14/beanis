# AGENTS.md

Notes for coding agents working in this repository. For using Beanis in other projects, read `plugins/beanis/skills/beanis/SKILL.md` (the API reference agents should follow) or `llms.txt`.

## Layout

- `beanis/odm/documents.py`: `Document` (CRUD, `find`, `all`, `vector_search`, TTL, field operations).
- `beanis/odm/indexes.py`: `Indexed`, `IndexedField`, `VectorField`, `GeoPoint`, `IndexManager` (set, sorted-set, geo and vector-set indexes).
- `beanis/odm/utils/init.py`: `init_beanis`; checks for Redis 8 vector sets when a model has vector fields.
- `beanis/memory.py`: `Memory`, `MemoryRecord`, `fastembed_embedder`, `memory_from_env`.
- `beanis/cli.py`: the `beanis` command (doctor, memory, mcp). JSON on stdout only.
- `beanis/mcp_server.py`: `beanis-mcp`, the stdio MCP server (needs the `mcp` extra, Python 3.10+).
- `plugins/beanis/` and `.claude-plugin/marketplace.json`: the Claude Code plugin (MCP server + skill).
- `docs/`: the documentation site. `docs/api/` is partly generated and still has pages from Beanie (queries, operators) that do not apply.

## Setup and tests

```bash
pip install -e ".[test,mcp]"
pytest                                   # fakeredis; real-Redis tests skip without a server
redis-server --port 6390 --save "" --daemonize yes    # Redis 8.0+
REDIS_PORT=6390 pytest tests/test_real_redis_regressions.py tests/test_memory_real_redis.py tests/test_readme_real_redis.py
```

fakeredis has no vector sets, so anything touching `VectorField`, `Memory`, the CLI memory commands or the MCP server must be tested against a real Redis 8 (`redis_client_real` fixture).

## Rules

- Beanis is not Beanie: do not add or document query-builder APIs (`to_list`, `find_one`, `sort`, `Model.field > x`). `find()` takes keyword filters on indexed fields.
- Every index change must keep `insert`, `save`, `update`, `delete_self`, `delete_many` and `delete_all` consistent (old entries removed, new ones added). Add a real-Redis regression test for index bugs.
- Test document models must be defined at module level.
- Keep the core dependency-free beyond `redis`, `pydantic`, `msgspec`, `lazy-model`; agent features go in the `mcp` extra and must import lazily.
- Python 3.9 syntax in the core package.
- Line length 79 (black/ruff config in `pyproject.toml`).
- When the public API changes, update `README.md`, `plugins/beanis/skills/beanis/SKILL.md`, `llms.txt` and `tests/test_readme_real_redis.py` together.
- Version lives in `pyproject.toml`, `beanis/__init__.py` and `plugins/beanis/.claude-plugin/plugin.json`.
