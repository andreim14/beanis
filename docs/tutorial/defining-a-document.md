# Defining a Document

The `Document` class in Beanis is responsible for mapping and handling data in Redis. It is inherited from the `BaseModel` Pydantic class, so it follows the same data typing and parsing behavior.

```python
from typing import Optional
from pydantic import BaseModel
from beanis import Document, Indexed


class Category(BaseModel):
    name: str
    description: str


class Product(Document):  # This is the model
    name: str
    description: Optional[str] = None
    price: Indexed(float)  # Indexed for range queries
    category: Category
    stock: int = 0

    class Settings:
        name = "products"  # Redis key prefix
```

## Fields

As mentioned before, the `Document` class is inherited from the Pydantic `BaseModel` class.
It uses all the same patterns of `BaseModel`. But also it has special types of fields:

- id
- Indexed

### id

The `id` field of the `Document` class reflects the unique identifier for the Redis document.
Each object of the `Document` type has this field.
The default type is `str` (UUID is auto-generated if not provided).

```python
class Sample(Document):
    num: int
    description: str

foo = await Sample.find(Sample.num > 5).first_or_none()

print(foo.id)  # This will print the id

bar = await Sample.get(foo.id)  # get by id
```

If you prefer another type, you can set it up too. For example, UUID:

```python
from uuid import UUID, uuid4
from pydantic import Field


class Sample(Document):
    id: UUID = Field(default_factory=uuid4)
    num: int
    description: str
```

### Indexed

To set up an index over a single field, the `Indexed` function can be used to wrap the type:

```python
from beanis import Document, Indexed


class Sample(Document):
    num: Indexed(int)  # Indexed for exact match queries
    price: Indexed(float)  # Indexed for range queries
    description: str
```

**How indexing works in Redis:**

- **Numeric fields** (int, float): Stored in Redis Sorted Sets for range queries
- **String/categorical fields**: Stored in Redis Sets for exact match queries

**Example queries:**

```python
# Range query on indexed field (uses Redis Sorted Set)
products = await Product.find(
    Product.price >= 10.0,
    Product.price <= 50.0
).to_list()

# Exact match on indexed field (uses Redis Set)
electronics = await Product.find(
    Product.category == "electronics"
).to_list()
```

## Settings

The inner `Settings` class is used to configure the document behavior:

```python
class Product(Document):
    name: str
    price: float

    class Settings:
        name = "products"  # Redis key prefix (default: class name)
        key_prefix = "prod"  # Alternative: custom prefix
        default_ttl = 3600  # Default TTL in seconds (optional)
        keep_nulls = False  # Don't store None values (default: True)
```

### Available Settings

- **name** or **key_prefix**: Redis key prefix (e.g., "Product:123")
- **default_ttl**: Default expiration time in seconds
- **keep_nulls**: Whether to store fields with None values
- **use_validation_on_fetch**: Validate data when reading from Redis (default: False for performance)

## Complex Types

Beanis automatically handles complex Pydantic types:

```python
from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime
from decimal import Decimal
from uuid import UUID


class Address(BaseModel):
    street: str
    city: str
    country: str


class Product(Document):
    name: str
    price: Decimal  # Precise decimal values
    tags: List[str]  # Lists
    metadata: Dict[str, str]  # Dictionaries
    category: Category  # Nested Pydantic models
    address: Optional[Address] = None  # Optional nested models
    created_at: datetime  # Datetime fields
    product_id: UUID  # UUID fields
```

All these types are automatically serialized to/from Redis!

## Custom Types

For types not natively supported (like NumPy arrays, PyTorch tensors), use custom encoders:

```python
from beanis import Document, register_type
import numpy as np
import base64
import pickle


# Register custom encoder/decoder
register_type(
    np.ndarray,
    encoder=lambda arr: base64.b64encode(pickle.dumps(arr)).decode(),
    decoder=lambda s: pickle.loads(base64.b64decode(s.encode()))
)


class MLModel(Document):
    name: str
    weights: Any  # Can store NumPy arrays!


model = MLModel(name="model_v1", weights=np.random.rand(100, 100))
await model.insert()  # NumPy array is automatically encoded
```

See the [Custom Encoders Guide](../../CUSTOM_ENCODERS.md) for more details.

## Document Storage

Beanis stores documents as **Redis Hashes**, which provides:

- Field-level access (`HGET`, `HSET` individual fields)
- Atomic increment/decrement operations
- Efficient memory usage
- Native Redis commands compatibility

**Example Redis structure for `Product:123`:**

```
HGETALL Product:123
{
    "name": "Tony's Chocolonely",
    "price": "5.95",
    "stock": "100",
    "category": "{\"name\":\"Chocolate\",\"description\":\"Roasted cacao\"}",
    "_class_name": "myapp.models.Product"
}
```

## Next Steps

- [Initialization](init.md) - Set up Beanis with Redis
- [Insert Operations](insert.md) - Create documents
- [Find Operations](find.md) - Query documents
- [Update Operations](update.md) - Modify documents
