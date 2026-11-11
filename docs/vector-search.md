# Vector search

Beanis stores embeddings in a **Redis 8 vector set** (`VADD` / `VSIM`), which ships with Redis 8.0 itself. There is no index to create and no module to install. Each document's short scalar fields travel with its vector, so a search can be filtered on them.

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

`embed` is any function from text to a list of floats; `beanis.memory.fastembed_embedder()` gives you a local one with no API key.

## Options

| Setting | Default | Notes |
|---|---|---|
| `dimensions` | required unless `None` | `None` takes the size of the first vector written |
| `quantization` | `"Q8"` | int8 vectors; `"NOQUANT"` keeps float32, `"BIN"` binary |
| metric | cosine | the only metric vector sets support |
| algorithm | HNSW | `m` and `ef_construction` are configurable |

## Filters

`filter` uses the vector-set expression syntax over the document's scalar fields (strings up to 256 characters, numbers, booleans):

```python
await Chunk.vector_search(vec, k=5, filter='.source == "faq" and .year >= 2025')
await Chunk.vector_search(vec, k=5, min_score=0.6)     # drop weak matches
```

`find_by_vector_similarity` returns `(id, score)` pairs without loading the documents.

## Keeping the index in sync

Inserting, saving, updating and deleting a document updates its vector. Documents that expire through a TTL are skipped at search time and their vectors removed.

## Requirements

Redis 8.0 or newer. On Redis 7, Redis Stack 7.x or Valkey, `init_beanis` raises at startup when a model has vector fields, instead of returning empty results. Run `beanis doctor` to check a server.
