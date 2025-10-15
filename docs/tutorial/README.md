# Beanis Tutorial

Welcome to the Beanis tutorial! This guide will help you get started with Beanis, a Redis ODM (Object-Document Mapper) for Python.

## What You'll Learn

This tutorial covers everything you need to build applications with Beanis and Redis:

- **Document modeling** - Define type-safe document structures
- **CRUD operations** - Create, read, update, and delete documents
- **Indexing & queries** - Fast queries using Redis Sorted Sets and Sets
- **Event hooks** - Run custom logic on document lifecycle events
- **Best practices** - Performance tips and patterns

## Tutorial Structure

### 1. Getting Started

Start here if you're new to Beanis:

#### [Defining a Document](defining-a-document.md)
Learn how to define document models with Pydantic:
- Basic document structure
- Field types and validation
- Indexed fields for queries
- Nested objects and complex types
- Custom encoders for special types (NumPy, PyTorch)

**Start here** → [defining-a-document.md](defining-a-document.md)

#### [Initialization](init.md)
Set up Beanis with your Redis client:
- Redis client configuration
- Initialize Beanis with document models
- FastAPI integration
- Multiple database support

**Next step** → [init.md](init.md)

### 2. Core Operations

Learn the essential document operations:

#### [Insert Documents](insert.md)
Create new documents in Redis:
- Insert single documents
- Bulk insert operations
- Insert with TTL (auto-expiration)
- Replace vs. insert behavior
- Event hooks on insert

**Learn inserting** → [insert.md](insert.md)

#### [Find Documents](find.md)
Query documents efficiently:
- Get by ID (O(1) lookup)
- Range queries on numeric fields
- Exact match on categorical fields
- Get all documents
- Pagination and sorting

**Learn querying** → [find.md](find.md)

#### [Update Documents](update.md)
Modify existing documents:
- Update specific fields
- Save entire document
- Atomic field operations (increment/decrement)
- Update with validation
- Event hooks on update

**Learn updating** → [update.md](update.md)

#### [Delete Documents](delete.md)
Remove documents from Redis:
- Delete single document
- Delete multiple documents
- Delete all documents
- TTL as alternative to deletion
- Event hooks on delete

**Learn deleting** → [delete.md](delete.md)

### 3. Advanced Topics

Master advanced Beanis features:

#### [Indexes](indexes.md)
Understand Redis indexing:
- Numeric indexes (Sorted Sets) for range queries
- Categorical indexes (Sets) for exact match
- Index creation and maintenance
- Query performance optimization
- Index limitations

**Learn indexing** → [indexes.md](indexes.md)

#### [Event Hooks (Actions)](actions.md)
Run custom logic on document lifecycle:
- Before/after insert hooks
- Before/after update hooks
- Before/after delete hooks
- Common patterns (timestamps, validation, audit logs)
- Hook execution order

**Learn hooks** → [actions.md](actions.md)

#### [Custom Encoders](custom-encoders.md)
Serialize complex Python types to Redis:
- NumPy arrays, PyTorch tensors
- Custom classes and dataclasses
- Binary data (images, audio)
- Performance optimization
- Versioning and error handling

**Learn custom encoders** → [custom-encoders.md](custom-encoders.md)

#### [Geo-Spatial Indexing](geo-spatial.md)
Build location-based features with sub-millisecond queries:
- Store/restaurant locators
- Delivery radius validation
- Real-time vehicle tracking
- Geo-fencing applications
- Complete delivery service example with benchmarks

**Learn geo-spatial** → [geo-spatial.md](geo-spatial.md)

## Quick Start Example

```python
from redis.asyncio import Redis
from beanis import Document, Indexed, init_beanis

# 1. Define a document
class Product(Document):
    name: str
    price: Indexed(float)  # Indexed for range queries
    category: Indexed(str)  # Indexed for exact match
    
    class Settings:
        name = "products"

# 2. Initialize
async def setup():
    client = Redis(decode_responses=True)
    await init_beanis(database=client, document_models=[Product])

# 3. Use it!
async def example():
    # Insert
    product = Product(name="Laptop", price=999.99, category="electronics")
    await product.insert()
    
    # Find by ID
    found = await Product.get(product.id)
    
    # Query by indexed field
    expensive = await Product.find(Product.price > 500).to_list()
    electronics = await Product.find(Product.category == "electronics").to_list()
    
    # Update
    await product.update(price=899.99)
    
    # Delete
    await product.delete_self()
```

## Recommended Learning Path

### For Beginners
1. [Defining a Document](defining-a-document.md) - Learn document structure
2. [Initialization](init.md) - Set up Beanis
3. [Insert Documents](insert.md) - Create documents
4. [Find Documents](find.md) - Query documents
5. [Update Documents](update.md) - Modify documents
6. [Delete Documents](delete.md) - Remove documents

### For Advanced Users
Start with the basics above, then explore:
7. [Indexes](indexes.md) - Optimize queries
8. [Event Hooks](actions.md) - Lifecycle events
9. [Custom Encoders](custom-encoders.md) - Complex type serialization
10. [Geo-Spatial Indexing](geo-spatial.md) - Location-based features

## Key Concepts

### Redis Storage Model

Beanis stores documents as **Redis Hashes**:
```
Product:prod_123 -> {name: "Laptop", price: "999.99", category: "electronics"}
```

### Indexing

Beanis automatically creates indexes for `Indexed()` fields:

- **Numeric fields** → Redis Sorted Set (range queries)
  ```
  idx:Product:price -> {prod_1: 999.99, prod_2: 1299.99, ...}
  ```

- **String fields** → Redis Set per value (exact match)
  ```
  idx:Product:category:electronics -> {prod_1, prod_3, ...}
  ```

### Type Safety

Beanis uses Pydantic for validation:
```python
class Product(Document):
    price: float  # Type-checked at runtime
    
# This raises ValidationError:
product = Product(name="Test", price="invalid")  # ❌
```

## Differences from MongoDB/Beanie

If you're coming from MongoDB/Beanie, note these differences:

| Feature | MongoDB/Beanie | Beanis (Redis) |
|---------|----------------|----------------|
| Storage | Documents (BSON) | Hashes (key-value) |
| Queries | Full query language | Indexed fields only |
| Relations | Link, BackLink | Use embedded documents |
| Aggregations | Pipeline | Not supported |
| Transactions | Multi-document | Single document |
| Full-text search | Text indexes | Use RediSearch module |

**Best practice**: Use embedded Pydantic models instead of document relations.

## Performance Tips

1. **Use batch operations** - `insert_many()`, `get_many()`, `delete_many()`
2. **Index selectively** - Only fields you'll query frequently
3. **Use TTL** - Automatically expire temporary data
4. **Leverage atomic operations** - `increment_field()` for counters
5. **Profile with realistic data** - Test performance at scale

## Need Help?

- **Documentation**: [Main Docs](../index.md)
- **Getting Started**: [Getting Started Guide](../getting-started.md)
- **GitHub Issues**: [Report bugs or request features](https://github.com/andreim14/beanis/issues)
- **Examples**: Check the tests folder for more examples

## What's Not Covered

Features not supported in the Redis version:

- ❌ **Relations** (Link/BackLink) - Use embedded documents
- ❌ **Migrations** - Not needed (schema-less)
- ❌ **Aggregations** - Use Python for data processing
- ❌ **Views** - Not applicable to Redis
- ❌ **Time Series** - Use Redis TimeSeries module or TTL

## Next Steps

Ready to start? Begin with [Defining a Document](defining-a-document.md)!

Already familiar with the basics? Jump to [Indexes](indexes.md) or [Event Hooks](actions.md) for advanced features.

Happy coding! 🚀
