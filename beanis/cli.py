"""
beanis command line: check a Redis server, use agent memory, run the MCP server.

    beanis doctor
    beanis memory remember "Deploys go out on Tuesdays" --user alice
    beanis memory recall "when do we deploy?" --user alice
    beanis memory recent
    beanis memory forget <id>
    beanis mcp

Every command prints one JSON value on stdout, so agents and scripts can
parse it. Errors go to stderr with a non-zero exit code.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional

import beanis

REDIS_URL_DEFAULT = "redis://localhost:6379/0"


def _out(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _record(record: Any, score: Optional[float] = None) -> Dict[str, Any]:
    data = {
        "id": record.id,
        "text": record.text,
        "kind": record.kind,
        "user_id": record.user_id,
        "session_id": record.session_id,
        "created_at": record.created_at,
    }
    if score is not None:
        data["score"] = round(score, 4)
    return data


def _check(name: str, ok: bool, detail: str, fix: str = "") -> Dict[str, Any]:
    return {
        "name": name,
        "ok": ok,
        "detail": detail,
        **({"fix": fix} if fix and not ok else {}),
    }


def _pairs(reply: Any) -> Dict[str, Any]:
    """VINFO reply: a map (RESP3) or a flat [key, value, ...] list (RESP2)."""
    if isinstance(reply, dict):
        return {str(k): v for k, v in reply.items()}
    return {str(k): v for k, v in zip(reply[0::2], reply[1::2])}


def _embed_model_dim(model_name: str) -> Optional[int]:
    """Output size of a fastembed model, from its catalog (no download)."""
    from fastembed import TextEmbedding

    for entry in TextEmbedding.list_supported_models():
        if entry.get("model") == model_name:
            return entry.get("dim")
    return None


async def doctor(url: str) -> Dict[str, Any]:
    from redis.asyncio import Redis

    from beanis.memory import DEFAULT_EMBED_MODEL, MemoryRecord
    from beanis.odm.indexes import IndexManager

    checks: List[Dict[str, Any]] = []
    checks.append(
        _check(
            "python",
            sys.version_info >= (3, 9),
            sys.version.split()[0],
            "Beanis needs Python 3.9+",
        )
    )
    has_fastembed = importlib.util.find_spec("fastembed") is not None
    has_mcp = importlib.util.find_spec("mcp") is not None
    extras_fix = 'pip install "beanis[mcp]" (needs Python 3.10+)'
    checks.append(
        _check(
            "fastembed",
            has_fastembed,
            "installed" if has_fastembed else "missing",
            extras_fix,
        )
    )
    checks.append(
        _check(
            "mcp", has_mcp, "installed" if has_mcp else "missing", extras_fix
        )
    )

    client = Redis.from_url(url, decode_responses=True)
    try:
        await client.ping()
    except Exception as e:  # connection refused, auth, DNS, ...
        checks.append(
            _check(
                "redis",
                False,
                f"cannot connect to {url}: {e}",
                "Start Redis 8: docker run -d -p 6379:6379 redis:8 (or brew install redis), or set REDIS_URL",
            )
        )
        return {"ok": False, "checks": checks}

    try:
        info = await client.info("server")
        version = str(info.get("redis_version", "?"))
        checks.append(_check("redis", True, f"{url} (Redis {version})"))

        vadd = await client.execute_command("COMMAND", "INFO", "VADD")
        has_vsets = bool(
            vadd and (vadd[0] if isinstance(vadd, list) else vadd)
        )
        checks.append(
            _check(
                "vector sets",
                has_vsets,
                (
                    "VADD/VSIM available"
                    if has_vsets
                    else f"Redis {version} has no VADD"
                ),
                "Vector search and memory need Redis 8.0+ (Valkey and Redis Stack 7.x lack vector sets)",
            )
        )

        if has_vsets:
            key = IndexManager.get_index_key(MemoryRecord, "embedding")
            model = os.environ.get("BEANIS_EMBED_MODEL", DEFAULT_EMBED_MODEL)
            if await client.exists(key):
                vinfo = _pairs(await client.execute_command("VINFO", key))
                stored = int(vinfo.get("vector-dim", 0))
                count = int(vinfo.get("size", 0))
                expected = _embed_model_dim(model) if has_fastembed else None
                ok = expected is None or expected == stored
                checks.append(
                    _check(
                        "memory",
                        ok,
                        f"{count} memories, {stored}-dim vectors"
                        + (f"; {model} gives {expected}" if expected else ""),
                        f"Stored memories were embedded with a {stored}-dim model; set BEANIS_EMBED_MODEL to that model or clear {key}",
                    )
                )
            else:
                checks.append(
                    _check("memory", True, "empty (created on first remember)")
                )
    finally:
        await client.aclose()

    return {"ok": all(c["ok"] for c in checks), "checks": checks}


async def memory_command(args: argparse.Namespace) -> Any:
    from beanis.memory import memory_from_env

    memory = memory_from_env(namespace=args.namespace)
    try:
        if args.action == "remember":
            record = await memory.remember(
                args.text,
                user_id=args.user,
                session_id=args.session,
                kind=args.kind,
                ttl=args.ttl,
            )
            return _record(record)
        if args.action == "recall":
            hits = await memory.recall(
                args.query,
                k=args.k,
                user_id=args.user,
                session_id=args.session,
                kind=args.kind,
                min_score=args.min_score,
            )
            return [_record(r, s) for r, s in hits]
        if args.action == "forget":
            return {"id": args.id, "deleted": await memory.forget(args.id)}
        records = await memory.recent(
            user_id=args.user,
            session_id=args.session,
            kind=args.kind,
            limit=args.limit,
        )
        return [_record(r) for r in records]
    finally:
        await memory.redis.aclose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beanis",
        description="Beanis: Redis ODM with vector search and agent memory. Output is JSON.",
    )
    parser.add_argument(
        "--version", action="version", version=f"beanis {beanis.__version__}"
    )
    parser.add_argument(
        "--url",
        help=f"Redis URL (default: $REDIS_URL or {REDIS_URL_DEFAULT})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "doctor", help="check Redis, vector sets, extras and stored memory"
    )
    sub.add_parser(
        "mcp", help="run the memory MCP server on stdio (same as beanis-mcp)"
    )

    mem = sub.add_parser(
        "memory", help="long-term memory: remember, recall, forget, recent"
    )
    mem.add_argument(
        "--namespace", help="default: $BEANIS_MEMORY_NAMESPACE or 'default'"
    )
    actions = mem.add_subparsers(dest="action", required=True)

    def scope(
        p: argparse.ArgumentParser, kind_default: Optional[str] = None
    ) -> None:
        p.add_argument("--user", help="user id")
        p.add_argument("--session", help="session id")
        p.add_argument(
            "--kind",
            default=kind_default,
            help="kind, e.g. note, preference, decision",
        )

    p = actions.add_parser(
        "remember", help="store a memory (near-duplicates are merged)"
    )
    p.add_argument("text")
    scope(p, kind_default="note")
    p.add_argument("--ttl", type=int, help="expire after this many seconds")

    p = actions.add_parser(
        "recall", help="memories most relevant to a query, best first"
    )
    p.add_argument("query")
    scope(p)
    p.add_argument("-k", type=int, default=5, help="how many (default 5)")
    p.add_argument(
        "--min-score", type=float, default=0.0, help="similarity floor, 0-1"
    )

    p = actions.add_parser("forget", help="delete one memory by id")
    p.add_argument("id")

    p = actions.add_parser("recent", help="newest memories first")
    scope(p)
    p.add_argument("--limit", type=int, default=10)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.url:
        os.environ["REDIS_URL"] = args.url
    url = os.environ.get("REDIS_URL", REDIS_URL_DEFAULT)

    try:
        if args.command == "mcp":
            from beanis.mcp_server import main as mcp_main

            mcp_main()
            return 0
        if args.command == "doctor":
            report = asyncio.run(doctor(url))
            _out(report)
            return 0 if report["ok"] else 1
        _out(asyncio.run(memory_command(args)))
        return 0
    except ImportError as e:
        print(f"beanis: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"beanis: {type(e).__name__}: {e}", file=sys.stderr)
        print(
            "beanis: run `beanis doctor` to check the setup", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
