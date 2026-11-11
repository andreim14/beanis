"""
MCP server that gives any agent a long-term memory backed by Redis 8.

    pip install "beanis[mcp]"
    REDIS_URL=redis://localhost:6379/0 beanis-mcp

Tools: remember, recall, forget, recent. Embeddings are computed locally with
fastembed (BAAI/bge-small-en-v1.5 by default), so no API key is needed.

Environment:
    REDIS_URL                  default redis://localhost:6379/0
    BEANIS_MEMORY_NAMESPACE    default "default"
    BEANIS_EMBED_MODEL         default BAAI/bge-small-en-v1.5
"""

from typing import List, Optional

from beanis.memory import Memory, memory_from_env

try:
    from mcp.server.mcpserver import MCPServer
except ImportError as e:  # pragma: no cover
    raise ImportError('The MCP server needs: pip install "beanis[mcp]"') from e

server = MCPServer(
    name="beanis-memory",
    instructions=(
        "Long-term memory. Call recall before answering questions that may depend "
        "on what the user told you earlier; call remember when the user shares a "
        "durable fact, preference or decision. Scope with user_id when you know it."
    ),
)

_memory: Optional[Memory] = None


async def _get_memory() -> Memory:
    global _memory
    if _memory is None:
        _memory = await memory_from_env().setup()
    return _memory


@server.tool()
async def remember(
    text: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    kind: str = "note",
) -> dict:
    """Store a fact, preference or decision worth keeping across conversations."""
    record = await (await _get_memory()).remember(
        text, user_id=user_id, session_id=session_id, kind=kind
    )
    return {"id": record.id, "text": record.text}


@server.tool()
async def recall(
    query: str,
    k: int = 5,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    kind: Optional[str] = None,
) -> List[dict]:
    """Return the stored memories most relevant to the query, best first."""
    hits = await (await _get_memory()).recall(
        query, k=k, user_id=user_id, session_id=session_id, kind=kind
    )
    return [
        {"id": r.id, "text": r.text, "kind": r.kind, "score": round(s, 4)}
        for r, s in hits
    ]


@server.tool()
async def forget(memory_id: str) -> bool:
    """Delete one memory by id (as returned by recall or remember)."""
    return await (await _get_memory()).forget(memory_id)


@server.tool()
async def recent(
    user_id: Optional[str] = None, kind: Optional[str] = None, limit: int = 10
) -> List[dict]:
    """List the newest memories, optionally for one user or kind."""
    records = await (await _get_memory()).recent(
        user_id=user_id, kind=kind, limit=limit
    )
    return [{"id": r.id, "text": r.text, "kind": r.kind} for r in records]


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
