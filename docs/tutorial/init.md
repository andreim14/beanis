# Initialization

Before you can use Beanis, you need to initialize it with your Redis client and document models.

## Basic Initialization

```python
import asyncio
from redis.asyncio import Redis
from beanis import Document, init_beanis


class Product(Document):
    name: str
    price: float

    class Settings:
        name = "products"


async def init():
    # Create Redis client
    # IMPORTANT: decode_responses=True is required!
    client = Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True
    )

    # Initialize Beanis
    await init_beanis(database=client, document_models=[Product])


if __name__ == "__main__":
    asyncio.run(init())
```

## Multiple Document Models

You can register multiple document models at once:

```python
from beanis import init_beanis


class Product(Document):
    name: str
    price: float


class User(Document):
    name: str
    email: str


class Order(Document):
    user_id: str
    product_ids: List[str]


await init_beanis(
    database=client,
    document_models=[Product, User, Order]
)
```

## Redis Client Configuration

### Basic Configuration

```python
from redis.asyncio import Redis

client = Redis(
    host="localhost",
    port=6379,
    db=0,  # Redis database number (0-15)
    decode_responses=True,  # Required!
    password="your-password",  # If Redis has auth
)
```

### Connection Pool

For production, use a connection pool:

```python
from redis.asyncio import Redis, ConnectionPool

pool = ConnectionPool(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True,
    max_connections=50,
)

client = Redis(connection_pool=pool)
```

### Redis URL

You can also use a connection URL:

```python
client = Redis.from_url(
    "redis://localhost:6379/0",
    decode_responses=True
)
```

## Application Integration

### FastAPI Example

```python
from fastapi import FastAPI
from redis.asyncio import Redis
from beanis import init_beanis
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    client = Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True
    )
    await init_beanis(database=client, document_models=[Product, User])

    yield

    # Shutdown
    await client.close()


app = FastAPI(lifespan=lifespan)


@app.get("/products")
async def get_products():
    products = await Product.all()
    return products
```

## Important Notes

1. **Always use `decode_responses=True`** - Beanis requires string responses from Redis
2. **Initialize once** - Call `init_beanis` once at application startup
3. **Close connections** - Always close the Redis client on shutdown

## Next Steps

- [Defining Documents](defining-a-document.md) - Learn about document models
- [Insert Operations](insert.md) - Create documents
- [Find Operations](find.md) - Query documents
