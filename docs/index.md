[![Beanis](https://raw.githubusercontent.com/andreim14/beanis/main/assets/logo/logo-no-background.svg)](https://github.com/andreim14/beanis)

<div align="center">
  <a href="https://pypi.python.org/pypi/beanis"><img src="https://img.shields.io/pypi/v/beanis" alt="PyPI version"></a>
</div>

## Overview

[Beanis](https://github.com/andreim14/beanis) - **"Beanie for Redis"** - is an asynchronous Python object-document mapper (ODM) for Redis. Data models are based on [Pydantic](https://pydantic-docs.helpmanual.io/).

When using Beanis each document has a corresponding `Document` class that is used to interact with Redis. In addition to retrieving data, Beanis allows you to add, update, or delete documents as well.

Beanis saves you time by removing boilerplate code, and it helps you focus on the parts of your app that actually matter.

**Works with vanilla Redis** - no RedisJSON or RediSearch modules required!

## Installation

### PIP

```shell
pip install beanis
```

### Poetry

```shell
poetry add beanis
```

## Quick Example

```python
import asyncio
from typing import Optional
from redis.asyncio import Redis
from pydantic import BaseModel
from beanis import Document, init_beanis


class Category(BaseModel):
    name: str
    description: str


class Product(Document):
    name: str  # You can use normal types just like in pydantic
    description: Optional[str] = None
    price: float
    category: Category  # You can include pydantic models as well
    stock: int = 0

    class Settings:
        name = "products"


# This is an asynchronous example, so we will access it from an async function
async def example():
    # Beanis uses Redis async client
    client = Redis(host="localhost", port=6379, db=0, decode_responses=True)

    # Initialize beanis with the Product document class
    await init_beanis(database=client, document_models=[Product])

    chocolate = Category(
        name="Chocolate",
        description="A preparation of roasted and ground cacao seeds."
    )

    # Beanis documents work just like pydantic models
    product = Product(name="Tony's Chocolonely", price=5.95, category=chocolate, stock=100)

    # And can be inserted into Redis
    await product.insert()

    # You can retrieve documents by ID
    found = await Product.get(product.id)

    # Update documents
    await product.update(price=6.95, stock=150)

    # Get all products
    all_products = await Product.all()

    # Clean up
    await client.close()


if __name__ == "__main__":
    asyncio.run(example())
```

## Key Features

- ✅ **Beanie-like API** - Familiar interface for MongoDB/Beanie developers
- ✅ **Works with vanilla Redis** - No modules required
- ✅ **Type Safety** - Full Pydantic validation
- ✅ **Fast** - Only 8% overhead vs raw Redis
- ✅ **Custom Encoders** - Store NumPy arrays, PyTorch tensors, any type
- ✅ **TTL Support** - Built-in expiration
- ✅ **Batch Operations** - Efficient pipelines
- ✅ **Event Hooks** - Before/after insert, update, delete

## Documentation

For detailed documentation, visit the main [README](../README.md) or check out:

- [Custom Encoders Guide](../CUSTOM_ENCODERS.md) - Store any Python type
- [Getting Started](getting-started.md) - Complete tutorial
- [Tutorial](tutorial/defining-a-document.md) - Step-by-step guides

## Credits

Beanis is inspired by [Beanie](https://github.com/BeanieODM/beanie) - the amazing MongoDB ODM.

We took the Beanie philosophy and adapted it for Redis, creating a simple yet powerful ODM that works with vanilla Redis.

## License

Apache License 2.0
