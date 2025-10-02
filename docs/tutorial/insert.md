# Insert Documents

Beanis documents behave just like Pydantic models (because they subclass `pydantic.BaseModel`). Hence, a document can be created in a similar fashion to Pydantic:

```python
from typing import Optional
from pydantic import BaseModel
from beanis import Document


class Category(BaseModel):
    name: str
    description: str


class Product(Document):
    name: str
    description: Optional[str] = None
    price: float
    category: Category

    class Settings:
        name = "products"


# Create a product
chocolate = Category(
    name="Chocolate",
    description="A preparation of roasted and ground cacao seeds."
)

product = Product(
    name="Tony's Chocolonely",
    description="Awesome chocolate bar",
    price=5.95,
    category=chocolate
)
```

## Insert One

To insert the document into Redis, use the `insert()` method:

```python
await product.insert()
print(product.id)  # Auto-generated UUID
```

The document is now stored in Redis as a Hash at key `Product:{id}`.

### Insert with Custom ID

You can specify a custom ID:

```python
product = Product(
    id="prod_001",
    name="Tony's Chocolonely",
    price=5.95,
    category=chocolate
)
await product.insert()
```

### Insert with TTL

Documents can expire automatically using TTL (time-to-live):

```python
# Expire after 1 hour
await product.insert(ttl=3600)
```

## Insert Many

For bulk inserts, use `insert_many()` for better performance:

```python
products = [
    Product(name=f"Product {i}", price=float(i * 10), category=chocolate)
    for i in range(100)
]

await Product.insert_many(products)
```

This uses Redis pipelines for efficient batch operations.

## Response

The `insert()` method returns the document with the populated `id` field:

```python
product = Product(name="New Product", price=9.99, category=chocolate)
result = await product.insert()

print(result.id)  # UUID automatically generated
print(product.id)  # Same as result.id
```

## Replace on Insert

By default, if a document with the same ID already exists, `insert()` will overwrite it. This is different from MongoDB's behavior.

```python
# First insert
product = Product(id="prod_001", name="Original", price=10.0, category=chocolate)
await product.insert()

# Second insert with same ID - replaces the first
product = Product(id="prod_001", name="Updated", price=15.0, category=chocolate)
await product.insert()  # Replaces the original
```

To check if a document exists first:

```python
if not await Product.exists("prod_001"):
    await product.insert()
```

## Event Hooks

You can use event hooks to run code before/after inserts:

```python
from beanis import Document, before_event, after_event, Insert


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


product = Product(name="Test", price=9.99)
await product.insert()  # Runs validate_price, then insert, then log_creation
```

## Examples

### Insert with Validation

```python
from pydantic import field_validator


class Product(Document):
    name: str
    price: float
    stock: int

    @field_validator('price')
    @classmethod
    def price_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Price must be positive')
        return v


product = Product(name="Test", price=-10, stock=100)
await product.insert()  # Raises ValidationError
```

### Insert with Complex Types

```python
from typing import List
from datetime import datetime


class Product(Document):
    name: str
    price: float
    tags: List[str]
    created_at: datetime

    class Settings:
        name = "products"


product = Product(
    name="Laptop",
    price=999.99,
    tags=["electronics", "computers"],
    created_at=datetime.now()
)
await product.insert()
```

### Bulk Insert with Error Handling

```python
products = [Product(name=f"Product {i}", price=float(i)) for i in range(100)]

try:
    await Product.insert_many(products)
    print(f"Inserted {len(products)} products")
except Exception as e:
    print(f"Error inserting products: {e}")
```

## Performance Tips

1. **Use `insert_many()` for bulk operations** - Much faster than individual inserts
2. **Set appropriate TTL** - Automatically clean up old data
3. **Use pipelines** - `insert_many()` already does this
4. **Avoid validation on insert if data is trusted** - Pydantic validation can be expensive

## Next Steps

- [Find Operations](find.md) - Query documents
- [Update Operations](update.md) - Modify documents
- [Delete Operations](delete.md) - Remove documents
