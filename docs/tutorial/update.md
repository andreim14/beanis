# Update Documents

Beanis provides several ways to update documents in Redis.

## Update Specific Fields

Use the `update()` method to modify specific fields:

```python
from beanis import Document, Indexed

class Product(Document):
    name: str
    price: Indexed(float)
    stock: int
    
    class Settings:
        name = "products"

# Get a product
product = await Product.get("product_id_123")

# Update specific fields
await product.update(
    price=7.99,
    stock=50
)

# Other fields remain unchanged
print(product.name)  # Original value preserved
```

## Save Method

The `save()` method inserts new documents or updates existing ones:

```python
# Create new document
product = Product(name="New Product", price=9.99, stock=10)
await product.save()  # Inserts

# Modify and save again
product.price = 12.99
await product.save()  # Updates
```

The `save()` method replaces the entire document in Redis.

## Update Fields Directly

Modify model attributes and call `update()`:

```python
product = await Product.get("product_id_123")

# Modify attributes
product.price = 15.99
product.stock = 100

# Save changes
await product.update(price=product.price, stock=product.stock)
```

## Atomic Field Operations

### Increment/Decrement Numeric Fields

Use `increment_field()` for atomic numeric updates:

```python
# Decrement stock atomically (thread-safe)
new_stock = await product.increment_field("stock", -1)
print(f"Stock after sale: {new_stock}")

# Increment by positive value
new_stock = await product.increment_field("stock", 10)
print(f"Stock after restock: {new_stock}")
```

This uses Redis `HINCRBY` for atomic operations - perfect for inventory management, counters, etc.

### Set Single Field

Update a single field without loading the entire document:

```python
# Update just the price field
await product.set_field("price", 8.99)

# Get just the price field
price = await product.get_field("price")
print(f"Current price: {price}")
```

This is efficient when you only need to modify one field.

## Update Multiple Documents

Update several documents by ID:

```python
# Update product prices
products = await Product.get_many(["id1", "id2", "id3"])

for product in products:
    if product:
        await product.update(price=product.price * 1.1)  # 10% increase
```

For better performance with many updates, use Redis pipelines:

```python
# More efficient for bulk updates
products = await Product.get_many(product_ids)

# Prepare updates
updates = []
for product in products:
    if product:
        product.price = product.price * 1.1
        updates.append(product)

# Execute in batch (uses pipeline internally)
for product in updates:
    await product.update(price=product.price)
```

## Update with Validation

Updates trigger Pydantic validation:

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

product = await Product.get("product_id_123")

# This will raise ValidationError
await product.update(price=-5.0)  # ❌ ValidationError!
```

## Update with Event Hooks

Run custom logic before/after updates:

```python
from beanis import before_event, after_event, Update

class Product(Document):
    name: str
    price: float
    old_price: float = 0.0
    
    @before_event(Update)
    async def store_old_price(self):
        # Get current price before update
        existing = await Product.get(self.id)
        if existing:
            self.old_price = existing.price
    
    @after_event(Update)
    async def log_price_change(self):
        if self.old_price != self.price:
            print(f"Price changed: ${self.old_price} -> ${self.price}")

product = await Product.get("product_id_123")
await product.update(price=7.99)
# Output: Price changed: $5.99 -> $7.99
```

## Update with TTL

Update and set/modify expiration:

```python
# Update with new TTL
await product.update(price=9.99)
await product.set_ttl(3600)  # Expire in 1 hour

# Check TTL
ttl = await product.get_ttl()
print(f"Expires in {ttl} seconds")

# Remove TTL (make permanent)
await product.persist()
```

## Update Nested Objects

Beanis stores nested Pydantic models as JSON strings in Redis:

```python
from pydantic import BaseModel

class Category(BaseModel):
    name: str
    description: str

class Product(Document):
    name: str
    price: float
    category: Category
    
    class Settings:
        name = "products"

product = await Product.get("product_id_123")

# Update nested object (replaces entire category)
await product.update(
    category=Category(name="Electronics", description="Electronic devices")
)

# To modify nested field, recreate the object
new_category = product.category
new_category.description = "Updated description"
await product.update(category=new_category)
```

## Important Notes

1. **update() replaces fields** - Only specified fields are updated, others remain unchanged
2. **save() replaces entire document** - All fields are overwritten
3. **Atomic operations** - Use `increment_field()` for thread-safe numeric updates
4. **Validation always runs** - Updates trigger Pydantic validation
5. **TTL is preserved** - Unless explicitly changed, document TTL remains unchanged

## Performance Tips

1. **Use increment_field() for counters** - Atomic and efficient
2. **Batch updates when possible** - Reduces Redis round trips
3. **Update only changed fields** - Don't call update() with all fields
4. **Use set_field() for single fields** - More efficient than loading entire document

## Examples

### Inventory Management

```python
async def process_sale(product_id: str, quantity: int):
    """Atomic stock update for sale"""
    product = await Product.get(product_id)
    
    # Check stock
    if product.stock < quantity:
        raise ValueError("Insufficient stock")
    
    # Atomic decrement
    new_stock = await product.increment_field("stock", -quantity)
    
    return new_stock

# Process a sale
remaining = await process_sale("prod_123", quantity=5)
print(f"Stock remaining: {remaining}")
```

### Price Updates with History

```python
from datetime import datetime
from typing import List

class PriceHistory(BaseModel):
    price: float
    timestamp: datetime

class Product(Document):
    name: str
    price: float
    price_history: List[PriceHistory] = []
    
    async def update_price(self, new_price: float):
        """Update price and maintain history"""
        # Add current price to history
        self.price_history.append(
            PriceHistory(price=self.price, timestamp=datetime.now())
        )
        
        # Update to new price
        await self.update(
            price=new_price,
            price_history=self.price_history
        )

product = await Product.get("prod_123")
await product.update_price(7.99)
```

### Conditional Updates

```python
async def apply_discount(product_id: str, discount_pct: float):
    """Apply discount only if price is above threshold"""
    product = await Product.get(product_id)
    
    if product.price >= 10.0:
        new_price = product.price * (1 - discount_pct / 100)
        await product.update(price=round(new_price, 2))
        return True
    
    return False

# Apply 20% discount to expensive items
discounted = await apply_discount("prod_123", 20.0)
```

## Next Steps

- [Delete Operations](delete.md) - Remove documents
- [Indexes](indexes.md) - Query optimization
- [Event Hooks](actions.md) - Document lifecycle events
