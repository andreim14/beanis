# Beanis - Redis ODM for Humans

[![Beanis](https://raw.githubusercontent.com/andreim14/beanis/main/assets/logo/logo-no-background.svg)](https://github.com/andreim14/beanis)

<div align="center">
  <a href="https://pypi.python.org/pypi/beanis"><img src="https://img.shields.io/pypi/v/beanis" alt="PyPI version"></a>
</div>

**Stop writing boilerplate Redis code. Focus on your application logic.**

Beanis is an async Python ODM (Object-Document Mapper) for Redis that gives you Pydantic models, type safety, and a clean API - while staying fast and working with vanilla Redis.

## Why Beanis?

### The Problem with Vanilla Redis

❌ **Manual serialization** - You write `json.dumps()` and `json.loads()` everywhere
❌ **Type conversions** - Strings from Redis need manual `float()`, `int()` conversions
❌ **Key management** - You track `"Product:123"`, `"all:Product"` keys manually
❌ **No validation** - Bad data silently corrupts your Redis database
❌ **Boilerplate code** - 15-20 lines for simple CRUD operations

### The Solution: Beanis

✅ **Automatic serialization** - Nested objects, lists, custom types - all handled
✅ **Type safety** - Full Pydantic validation + IDE autocomplete
✅ **Smart key management** - Focus on your data, not Redis internals
✅ **Data validation** - Catch errors before they hit Redis
✅ **Write 70% less code** - 5-7 lines for the same operations

**AND it's fast:** Only 8% overhead vs vanilla Redis

### Who Should Use Beanis?

✅ You're building a **production app** that needs Redis but not the boilerplate
✅ You want **type safety and validation** without sacrificing performance
✅ You're using **vanilla Redis** (no RedisJSON/RediSearch modules)
✅ You like **Beanie's MongoDB API** and want the same for Redis
✅ You're storing **complex data** (nested objects, NumPy arrays, etc.)

### When NOT to Use Beanis?

❌ You need every microsecond of performance (use raw redis-py)
❌ You need RedisJSON/RediSearch features (use Redis OM)
❌ You're only storing simple key-value pairs (use raw redis-py)

## Show Me The Code

### Basic CRUD Operation

<table>
<tr>
<th>Vanilla Redis (20 lines)</th>
<th>Beanis (7 lines)</th>
</tr>
<tr>
<td>

```python
import json
import time
from redis.asyncio import Redis

redis = Redis(decode_responses=True)

# Insert
product_data = {
    "name": "Tony's Chocolonely",
    "price": "5.95",
    "category": json.dumps({
        "name": "Chocolate",
        "description": "Roasted cacao"
    })
}
await redis.hset("Product:prod_123",
                 mapping=product_data)
await redis.zadd("all:Product",
                 {"prod_123": time.time()})

# Retrieve
raw = await redis.hgetall("Product:prod_123")
product = {
    "name": raw["name"],
    "price": float(raw["price"]),
    "category": json.loads(raw["category"])
}
```

</td>
<td>

```python
from beanis import Document
from pydantic import BaseModel

class Category(BaseModel):
    name: str
    description: str

class Product(Document):
    name: str
    price: float
    category: Category

# Insert
product = Product(
    name="Tony's Chocolonely",
    price=5.95,
    category=Category(
        name="Chocolate",
        description="Roasted cacao"
    )
)
await product.insert()

# Retrieve
found = await Product.get(product.id)
```

</td>
</tr>
</table>

**Result:** Type-safe, validated, **65% less code**

### Search/Query Operation

<table>
<tr>
<th>Vanilla Redis (25 lines)</th>
<th>Beanis (4 lines)</th>
</tr>
<tr>
<td>

```python
# Find products between $10-50
keys = await redis.zrangebyscore(
    "idx:Product:price",
    min=10.0,
    max=50.0
)

# Fetch each product using pipeline
pipe = redis.pipeline()
for key in keys:
    pipe.hgetall(f"Product:{key}")
results = await pipe.execute()

# Parse manually
products = []
for data in results:
    if data:
        products.append({
            "name": data["name"],
            "price": float(data["price"]),
            "stock": int(data["stock"]),
            "category": json.loads(
                data.get("category", "{}")
            )
        })
```

</td>
<td>

```python
# Find products between $10-50
products = await Product.find(
    Product.price >= 10.0,
    Product.price <= 50.0
).to_list()
```

</td>
</tr>
</table>

**Result:** **84% less code**, fully typed results

### Update Operation

<table>
<tr>
<th>Vanilla Redis (10 lines)</th>
<th>Beanis (6 lines)</th>
</tr>
<tr>
<td>

```python
# Update price and stock
await redis.hset("Product:123", mapping={
    "price": "6.95",
    "stock": "150"
})

# Atomic increment
new_stock = await redis.hincrby(
    "Product:123",
    "stock",
    -1
)
```

</td>
<td>

```python
# Update fields
await product.update(
    price=6.95,
    stock=150
)

# Atomic increment
new_stock = await product.increment_field(
    "stock", -1
)
```

</td>
</tr>
</table>

**Result:** Same functionality, cleaner API, type-safe

### Batch Operations

<table>
<tr>
<th>Vanilla Redis (14 lines)</th>
<th>Beanis (9 lines)</th>
</tr>
<tr>
<td>

```python
# Insert 100 products
pipe = redis.pipeline()
for i in range(100):
    product_id = f"prod_{i}"
    data = {
        "name": f"Product {i}",
        "price": str(i * 10),
        "stock": "100",
        "category": json.dumps({
            "name": "Category"
        })
    }
    pipe.hset(f"Product:{product_id}",
              mapping=data)
    pipe.zadd("all:Product",
              {product_id: time.time()})
await pipe.execute()
```

</td>
<td>

```python
# Insert 100 products
products = [
    Product(
        name=f"Product {i}",
        price=i * 10,
        stock=100,
        category=Category(name="Category")
    )
    for i in range(100)
]
await Product.insert_many(products)
```

</td>
</tr>
</table>

**Result:** **35% less code**, no manual key management

## Installation

### PIP

```shell
pip install beanis
```

### Poetry

```shell
poetry add beanis
```

## Quick Start

```python
import asyncio
from typing import Optional
from redis.asyncio import Redis
from pydantic import BaseModel
from beanis import Document, init_beanis, Indexed


class Category(BaseModel):
    name: str
    description: str


class Product(Document):
    name: str
    description: Optional[str] = None
    price: Indexed(float)  # Indexed for range queries
    category: Category
    stock: int = 0

    class Settings:
        name = "products"


async def main():
    # Initialize Redis client
    client = Redis(host="localhost", port=6379, db=0, decode_responses=True)

    # Initialize Beanis
    await init_beanis(database=client, document_models=[Product])

    # Create a product
    chocolate = Category(
        name="Chocolate",
        description="A preparation of roasted and ground cacao seeds."
    )

    product = Product(
        name="Tony's Chocolonely",
        price=5.95,
        category=chocolate,
        stock=100
    )

    # Insert into Redis
    await product.insert()

    # Retrieve by ID
    found = await Product.get(product.id)
    print(f"Found: {found.name} - ${found.price}")

    # Query by price range
    affordable = await Product.find(
        Product.price < 10.0
    ).to_list()
    print(f"Affordable products: {len(affordable)}")

    # Update specific fields
    await product.update(price=6.95, stock=150)

    # Atomic increment
    new_stock = await product.increment_field("stock", -1)
    print(f"Stock after sale: {new_stock}")

    # Get all products
    all_products = await Product.all()
    print(f"Total products: {len(all_products)}")

    # Delete
    await product.delete_self()

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## Core Features

### 🚀 Type Safety & Validation

Beanis uses Pydantic models, giving you automatic validation and type checking:

```python
class Product(Document):
    name: str
    price: float  # Must be a number
    stock: int    # Must be an integer
    category: Category  # Must be a valid Category object

# This will raise a validation error BEFORE hitting Redis
product = Product(
    name="Invalid",
    price="not a number",  # ❌ ValidationError!
    stock=100
)
```

### ⚡ High Performance

Beanis is optimized for speed with minimal overhead:

- **8% overhead** vs vanilla Redis (benchmarked)
- Uses `msgspec` for ultra-fast JSON parsing (2x faster than orjson)
- Skips Pydantic validation on reads by default (data from Redis is trusted)
- Efficient pipeline usage for batch operations

**Benchmark Results** (Get by ID):
- Vanilla Redis: 1.00x (baseline)
- Beanis: 1.08x (only 8% slower)
- Redis OM: 1.20x (20% slower)

See [FINAL_FAIR_BENCHMARK_RESULTS.md](FINAL_FAIR_BENCHMARK_RESULTS.md) for detailed benchmarks.

### 🎯 Pythonic API

Familiar Beanie-style interface for MongoDB developers:

```python
# Query with Pythonic operators
products = await Product.find(
    Product.price >= 10.0,
    Product.price <= 50.0,
    Product.stock > 0
).to_list()

# Chaining operations
expensive = await Product.find(
    Product.price > 100
).sort(Product.price).limit(10).to_list()

# Batch operations
await Product.insert_many([product1, product2, product3])
products = await Product.get_many([id1, id2, id3])
await Product.delete_many([id1, id2])
```

### 📦 Store Anything

Beanis handles complex types automatically:

**Built-in support:**
- Nested Pydantic models
- Lists, dicts, tuples, sets
- Decimal, UUID, Enum
- datetime, date, time, timedelta

**Custom types via encoders:**
```python
from beanis import Document, register_type
import numpy as np

# NumPy arrays work automatically (auto-registered)
class MLModel(Document):
    name: str
    weights: Any  # Stores np.ndarray!

model = MLModel(name="v1", weights=np.random.rand(100, 100))
await model.insert()  # Just works!

# Custom types
register_type(
    MyCustomType,
    encoder=lambda obj: str(obj),
    decoder=lambda s: MyCustomType.from_string(s)
)
```

See [CUSTOM_ENCODERS.md](CUSTOM_ENCODERS.md) for detailed documentation.

### 🔧 Production Ready Features

**TTL Support:**
```python
# Insert with TTL
await product.insert(ttl=3600)  # Expires in 1 hour

# Set TTL on existing document
await product.set_ttl(7200)
ttl = await product.get_ttl()
await product.persist()  # Remove TTL
```

**Event Hooks:**
```python
from beanis import before_event, after_event, Insert, Update

class Product(Document):
    name: str
    price: float

    @before_event(Insert)
    async def validate_price(self):
        if self.price < 0:
            raise ValueError("Price cannot be negative")

    @after_event(Insert)
    async def log_creation(self):
        print(f"Created product: {self.name}")
```

**Field-Level Operations:**
```python
# Get/set single field without loading entire document
price = await product.get_field("price")
await product.set_field("stock", 200)

# Atomic increment
new_stock = await product.increment_field("stock", 5)
```

**Document Tracking:**
```python
# Get all documents (sorted by insertion time)
all_products = await Product.all()

# Pagination
page1 = await Product.all(limit=10)
page2 = await Product.all(skip=10, limit=10)

# Count and delete all
count = await Product.count()
await Product.delete_all()
```

## Comparison

| Feature | Vanilla Redis | Beanis | Redis OM |
|---------|--------------|---------|----------|
| **Code volume** | 100% | **30%** ⭐ | 50% |
| **Type safety** | Manual | **Automatic** ⭐ | Automatic |
| **Performance** | **100%** ⭐ | 108% | 120% |
| **Vanilla Redis** | ✅ | **✅** ⭐ | ❌ Requires modules |
| **Validation** | Manual | **Automatic** ⭐ | Automatic |
| **API Style** | Redis commands | **Pythonic** ⭐ | Redis OM |
| **Learning curve** | Medium | **Easy** ⭐ | Medium |
| **Nested objects** | Manual | **Automatic** ⭐ | Automatic |
| **Custom types** | Manual | **Easy** ⭐ | Limited |
| **Event hooks** | ❌ | **✅** ⭐ | ❌ |
| **All DBs (0-15)** | ✅ | **✅** ⭐ | ❌ DB 0 only |

## Choosing the Right Tool

### Choose Vanilla Redis when:
- Every microsecond matters (high-frequency trading, etc.)
- Simple key-value storage
- You're a Redis expert and don't need abstractions

### Choose Beanis when: ⭐
- **Building production applications** with complex data models
- Want **type safety + performance** (8% overhead is acceptable)
- Using **vanilla Redis** (no RedisJSON/RediSearch modules)
- Need to store **nested objects, custom types, NumPy arrays**, etc.
- Coming from **MongoDB/Beanie** and want familiar API
- Want **event hooks** for validation and lifecycle management

### Choose Redis OM when:
- You need **RedisJSON/RediSearch** features
- Don't mind installing Redis modules
- Want Redis Stack integration
- Need advanced full-text search

## Documentation

- **[Custom Encoders Guide](CUSTOM_ENCODERS.md)** - Store NumPy, PyTorch, and custom types
- **[Complex Types Reference](COMPLEX_TYPES_SUPPORT.md)** - Built-in type support
- **[Performance Guide](FINAL_FAIR_BENCHMARK_RESULTS.md)** - Benchmarks and optimization

## Requirements

- Python 3.8+
- Redis 5.0+
- Pydantic 1.10+ or 2.0+

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=beanis

# Run specific test
pytest tests/test_core.py::test_insert_and_get
```

## Credits

Beanis is a fork of [Beanie](https://github.com/BeanieODM/beanie) - the amazing MongoDB ODM created by Roman Right and contributors.

We took the Beanie codebase and completely reimagined it for Redis, replacing MongoDB operations with Redis commands while preserving the elegant API design. If you're using MongoDB, check out the original [Beanie](https://github.com/BeanieODM/beanie) - it's awesome!

**Special thanks to:**
- Roman Right and the Beanie community for creating the foundation
- All Beanie contributors whose code inspired this project
- The Redis and Pydantic teams for their excellent libraries

[![Beanie](https://raw.githubusercontent.com/roman-right/beanie/main/assets/logo/white_bg.svg)](https://github.com/BeanieODM/beanie)

## License

Apache License 2.0
