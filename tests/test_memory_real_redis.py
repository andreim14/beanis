"""
Agent memory and MCP tools against a real Redis 8 server (REDIS_PORT).

Uses a deterministic bag-of-words embedder so results do not depend on a model.
"""
import hashlib
import math
import re

import pytest

from beanis import Memory

DIM = 64


def embed(text: str):
    vec = [0.0] * DIM
    for word in re.findall(r"[a-z]+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.mark.asyncio
async def test_recall_ranks_relevant_memory_first(redis_client_real):
    memory = Memory(redis_client_real, embed=embed, namespace="t1")
    await memory.remember("the user prefers dark mode in the editor", user_id="u1")
    await memory.remember("invoices are due on the fifth of each month", user_id="u1")

    hits = await memory.recall("dark mode editor preference", user_id="u1", k=2)

    assert hits[0][0].text.startswith("the user prefers dark mode")
    assert hits[0][1] > hits[1][1]


@pytest.mark.asyncio
async def test_recall_is_scoped_by_user_and_namespace(redis_client_real):
    mem_a = Memory(redis_client_real, embed=embed, namespace="a")
    mem_b = Memory(redis_client_real, embed=embed, namespace="b")
    await mem_a.remember("likes espresso", user_id="u1")
    await mem_a.remember("likes green tea", user_id="u2")
    await mem_b.remember("likes espresso a lot", user_id="u1")

    texts = [r.text for r, _ in await mem_a.recall("likes", user_id="u1", k=10)]

    assert texts == ["likes espresso"]


@pytest.mark.asyncio
async def test_near_duplicate_replaces_instead_of_adding(redis_client_real):
    memory = Memory(redis_client_real, embed=embed, namespace="dup")
    first = await memory.remember("meeting moved to friday", user_id="u1")
    second = await memory.remember("meeting moved to Friday", user_id="u1")

    assert first.id == second.id
    assert len(await memory.recent(user_id="u1")) == 1


@pytest.mark.asyncio
async def test_forget_removes_from_recall(redis_client_real):
    memory = Memory(redis_client_real, embed=embed, namespace="f")
    record = await memory.remember("temporary note about the demo", user_id="u1")

    assert await memory.forget(record.id) is True
    assert await memory.recall("demo note", user_id="u1") == []


@pytest.mark.asyncio
async def test_expired_memories_are_not_recalled(redis_client_real):
    memory = Memory(redis_client_real, embed=embed, namespace="ttl")
    record = await memory.remember("short lived reminder", user_id="u1", ttl=60)
    # Simulate expiry: the hash disappears, the vector-set entry is left behind
    await redis_client_real.delete(record._get_redis_key())

    assert await memory.recall("reminder", user_id="u1") == []


@pytest.mark.asyncio
async def test_mcp_tools(redis_client_real, monkeypatch):
    pytest.importorskip("mcp")
    from beanis import mcp_server

    monkeypatch.setattr(
        mcp_server,
        "_memory",
        await Memory(redis_client_real, embed=embed, namespace="mcp").setup(),
    )

    saved = await mcp_server.remember("deploys happen on tuesdays", user_id="u9")
    hits = await mcp_server.recall("when do deploys happen", user_id="u9")
    assert hits[0]["id"] == saved["id"] and 0 < hits[0]["score"] <= 1
    assert [m["id"] for m in await mcp_server.recent(user_id="u9")] == [saved["id"]]
    assert await mcp_server.forget(saved["id"]) is True
    assert await mcp_server.recall("deploys", user_id="u9") == []
