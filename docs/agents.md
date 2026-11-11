# Agent memory and MCP

`Memory` stores what an agent should keep between conversations and finds it again by meaning. Memories are scoped by namespace, user, session and kind; a new memory that is nearly identical to an old one replaces it instead of piling up. It needs Redis 8.0+.

There are four ways to use it: from Python, from the `beanis` command line, as an MCP server, and as a Claude Code plugin that loads each project's memory at the start of every session.

## From Python

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

`fastembed_embedder()` runs `BAAI/bge-small-en-v1.5` locally on the CPU, with no API key; it loads the model on first use. Any embedding function works, sync or async. Use the same model for every writer: all memories share one vector set.

## From the command line

`pip install "beanis[mcp]"` installs a `beanis` command. Every command except `context` prints JSON, so shell scripts and coding agents can use it directly.

```bash
beanis doctor                                   # checks Redis, vector sets, extras, stored memory
beanis memory remember "Deploys go out on Tuesdays" --kind decision
beanis memory recall "when do we deploy?" -k 3
beanis memory recent --limit 5
beanis memory forget <id>
beanis context                                  # newest memories as plain text, for session hooks
beanis mcp                                      # the MCP server on stdio
```

| Variable | Default | Meaning |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | server (or pass `--url`) |
| `BEANIS_MEMORY_NAMESPACE` | `default` | `auto` gives each git project its own namespace |
| `BEANIS_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model |

`beanis doctor` reports, among other things, when the stored memories were embedded with a model of a different size than the configured one.

## As an MCP server

`beanis-mcp` (or `beanis mcp`) serves the memory over MCP stdio with four tools: `remember`, `recall`, `forget` and `recent`.

=== "Claude Code"

    ```bash
    claude mcp add beanis-memory --env REDIS_URL=redis://localhost:6379/0 -- beanis-mcp
    ```

=== "Claude Desktop, Cursor and others"

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

## Claude Code plugin

The repository is also a Claude Code plugin marketplace:

```bash
claude plugin marketplace add andreim14/beanis
claude plugin install beanis@beanis
```

The plugin adds:

- the memory MCP server, with one namespace per git project;
- a SessionStart hook that puts the project's newest memories into Claude's context, so a new session does not start from zero;
- a skill that teaches Claude the Beanis API.

It starts everything with `uvx`, so it needs [uv](https://docs.astral.sh/uv/) and a Redis 8 at `REDIS_URL`.

!!! note "Why the session hook matters"
    In our tests, with memory offered only as MCP tools, Claude ignored them and kept notes in its own built-in memory. With the session hook, it stored a project briefing in the shared store unprompted, a fresh session answered from it, and parallel subagents read and wrote the same memories.

## What to store

Good memories are single, self-contained sentences that stay true for a while: decisions and their reasons, conventions, gotchas, environment facts, user preferences. Set `kind` to `decision`, `fact`, `preference` or `note`. Do not store secrets: memory is plain text in Redis, readable by anyone with access to the server.
