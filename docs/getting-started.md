# Getting Started

## Installing Beanis

You can simply install Beanis from [PyPI](https://pypi.org/project/beanis/):

### PIP

```shell
pip install beanis
```

### Poetry

```shell
poetry add beanis
```

## Initialization

Getting Beanis setup in your code is really easy:

1. Write your database model as a Pydantic class but use `beanis.Document` instead of `pydantic.BaseModel`
2. Initialize Redis async client
3. Call `beanis.init_beanis` with the Redis client and list of Beanis models

The code below should get you started and shows some of the field types that you can use with Beanis.

```python
import asyncio
from typing import Optional
from redis.asyncio import Redis
from pydantic import BaseModel
from beanis import Document, Indexed, init_beanis


class Category(BaseModel):
    name: str
    description: str


# This is the model that will be saved to Redis
class Product(Document):
    name: str  # You can use normal types just like in pydantic
    description: Optional[str] = None
    price: Indexed(float)  # Indexed for range queries
    category: Category  # You can include pydantic models as well
    stock: int = 0

    class Settings:
        name = "products"  # Redis key prefix


async def main():
    # Initialize Redis client
    # IMPORTANT: decode_responses=True is required!
    client = Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True
    )

    # Initialize Beanis with the Product document
    await init_beanis(database=client, document_models=[Product])

    # Now you can use your models!
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

    # Insert the document
    await product.insert()
    print(f"Inserted product with ID: {product.id}")

    # Retrieve by ID
    found = await Product.get(product.id)
    print(f"Found: {found.name} - ${found.price}")

    # Query by price
    affordable = await Product.find(Product.price < 10.0).to_list()
    print(f"Affordable products: {len(affordable)}")

    # Update
    await product.update(price=6.95, stock=150)
    print(f"Updated price to ${product.price}")

    # Get all products
    all_products = await Product.all()
    print(f"Total products: {len(all_products)}")

    # Delete
    await product.delete_self()
    print("Product deleted")

    # Clean up
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## What's Next?

Now that you have Beanis set up, you can explore:

- [Defining Documents](tutorial/defining-a-document.md) - Learn about document models
- [Insert Operations](tutorial/insert.md) - How to create documents
- [Find Operations](tutorial/find.md) - How to query documents
- [Update Operations](tutorial/update.md) - How to modify documents
- [Custom Encoders](../CUSTOM_ENCODERS.md) - Store NumPy arrays, custom types

## Key Differences from Beanie (MongoDB)

If you're coming from Beanie, here are the key differences:

1. **Redis client instead of Motor**: Use `redis.asyncio.Redis` instead of `motor.AsyncIOMotorClient`
2. **decode_responses=True required**: Redis needs this to return strings instead of bytes
3. **Hash storage**: Documents are stored as Redis Hashes, not MongoDB documents
4. **Indexed fields**: Uses Redis Sorted Sets/Sets for indexes, not MongoDB indexes
5. **No aggregation pipelines**: Use Redis native operations or query methods
6. **Custom encoders**: Built-in system for storing complex types like NumPy arrays

## Common Patterns

### Insert with TTL

```python
# Document expires after 1 hour
await product.insert(ttl=3600)
```

### Batch Operations

```python
# Insert many documents efficiently
products = [
    Product(name=f"Product {i}", price=float(i * 10))
    for i in range(100)
]
await Product.insert_many(products)

# Get many documents
ids = [p.id for p in products]
found = await Product.get_many(ids)
```

### Event Hooks

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

## Troubleshooting

### "All keys must be strings" error

Make sure you set `decode_responses=True` when creating the Redis client:

```python
client = Redis(decode_responses=True)  # ✅ Correct
client = Redis()  # ❌ Will cause errors
```

### Type validation errors

Beanis validates data using Pydantic. If you get validation errors, check that your data matches the model:

```python
# ❌ This will fail - price must be float
product = Product(name="Test", price="not a number")

# ✅ This works
product = Product(name="Test", price=9.99)
```

### Performance tips

1. Use batch operations (`insert_many`, `get_many`) for multiple documents
2. Skip validation on reads for better performance (default behavior)
3. Use TTL to automatically expire old data
4. Use indexed fields for range queries

## Next Steps

- Explore the [tutorial](tutorial/defining-a-document.md) for in-depth guides
- Check out [custom encoders](../CUSTOM_ENCODERS.md) for storing complex types
- Read the [README](../README.md) for feature comparisons
