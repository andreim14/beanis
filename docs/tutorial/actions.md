# Event Hooks (Actions)

Beanis provides event hooks that allow you to run custom logic before and after document operations.

## Overview

Event hooks enable you to:
- Validate data before saving
- Transform data before insertion
- Log operations
- Trigger side effects
- Maintain custom indexes
- Send notifications

## Available Events

Beanis supports hooks for these operations:

- `Insert` - When inserting new documents
- `Update` - When updating existing documents
- `Delete` - When deleting documents
- `Save` - When saving (insert or update)

## Basic Usage

### Before Event Hooks

Run logic **before** an operation:

```python
from beanis import Document, before_event, Insert

class Product(Document):
    name: str
    price: float
    
    @before_event(Insert)
    async def validate_price(self):
        """Validate price before inserting"""
        if self.price < 0:
            raise ValueError("Price cannot be negative")
        if self.price > 10000:
            raise ValueError("Price too high")

# This will raise ValueError
product = Product(name="Test", price=-5.0)
await product.insert()  # ❌ ValueError: Price cannot be negative
```

### After Event Hooks

Run logic **after** an operation:

```python
from beanis import Document, after_event, Insert
from datetime import datetime

class Product(Document):
    name: str
    price: float
    
    @after_event(Insert)
    async def log_creation(self):
        """Log after successful insert"""
        print(f"Created product '{self.name}' at {datetime.now()}")

product = Product(name="Laptop", price=999.99)
await product.insert()
# Output: Created product 'Laptop' at 2025-01-15 10:30:00
```

## Multiple Hooks

You can attach multiple hooks to the same event:

```python
from beanis import Document, before_event, after_event, Insert

class Product(Document):
    name: str
    price: float
    created_at: datetime = None
    
    @before_event(Insert)
    async def set_timestamp(self):
        """Set creation timestamp"""
        self.created_at = datetime.now()
    
    @before_event(Insert)
    async def validate_price(self):
        """Validate price"""
        if self.price < 0:
            raise ValueError("Price must be positive")
    
    @after_event(Insert)
    async def log_creation(self):
        """Log after insert"""
        print(f"Created: {self.name}")
    
    @after_event(Insert)
    async def send_notification(self):
        """Send notification"""
        # Send to analytics, message queue, etc.
        pass

# All hooks run in order
await product.insert()
```

**Execution order**: Hooks run in the order they're defined in the class.

## Hook Types

### Insert Hooks

Triggered when inserting new documents:

```python
from beanis import before_event, after_event, Insert

class Product(Document):
    name: str
    slug: str = ""
    
    @before_event(Insert)
    async def generate_slug(self):
        """Auto-generate slug from name"""
        self.slug = self.name.lower().replace(" ", "-")
    
    @after_event(Insert)
    async def index_for_search(self):
        """Add to external search index"""
        # Add to Elasticsearch, Algolia, etc.
        pass

product = Product(name="Tony's Chocolonely", price=5.95)
await product.insert()
print(product.slug)  # "tony's-chocolonely"
```

### Update Hooks

Triggered when updating existing documents:

```python
from beanis import before_event, after_event, Update

class Product(Document):
    name: str
    price: float
    updated_at: datetime = None
    
    @before_event(Update)
    async def set_updated_timestamp(self):
        """Update timestamp on every update"""
        self.updated_at = datetime.now()
    
    @after_event(Update)
    async def invalidate_cache(self):
        """Clear cache after update"""
        # Clear Redis cache, CDN, etc.
        pass

product = await Product.get("prod_123")
await product.update(price=7.99)
print(product.updated_at)  # 2025-01-15 10:35:00
```

### Save Hooks

Triggered by `save()` method (insert OR update):

```python
from beanis import before_event, after_event, Save

class Product(Document):
    name: str
    price: float
    
    @before_event(Save)
    async def validate_data(self):
        """Runs on both insert and update"""
        if self.price < 0:
            raise ValueError("Invalid price")
    
    @after_event(Save)
    async def log_save(self):
        """Log all saves"""
        print(f"Saved product {self.id}")

# Works for both
product = Product(name="New", price=9.99)
await product.save()  # Insert - hooks run

product.price = 12.99
await product.save()  # Update - hooks run again
```

### Delete Hooks

Triggered when deleting documents:

```python
from beanis import before_event, after_event, Delete

class Product(Document):
    name: str
    
    @before_event(Delete)
    async def backup_before_delete(self):
        """Backup before deletion"""
        # Save to backup storage
        print(f"Backing up product: {self.name}")
    
    @after_event(Delete)
    async def cleanup_resources(self):
        """Clean up related resources"""
        # Delete images, clear cache, etc.
        print(f"Cleaned up resources for {self.id}")

product = await Product.get("prod_123")
await product.delete_self()
# Output: Backing up product: Laptop
#         Cleaned up resources for prod_123
```

## Common Use Cases

### Auto-Timestamps

```python
from datetime import datetime
from beanis import Document, before_event, Insert, Update

class Product(Document):
    name: str
    created_at: datetime = None
    updated_at: datetime = None
    
    @before_event(Insert)
    async def set_created_at(self):
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    @before_event(Update)
    async def set_updated_at(self):
        self.updated_at = datetime.now()

product = Product(name="Test", price=9.99)
await product.insert()
print(product.created_at)  # 2025-01-15 10:00:00

await asyncio.sleep(2)
await product.update(price=12.99)
print(product.updated_at)  # 2025-01-15 10:00:02
```

### Slug Generation

```python
import re
from beanis import Document, before_event, Insert, Update

class Product(Document):
    name: str
    slug: str = ""
    
    @before_event(Insert, Update)
    async def generate_slug(self):
        """Auto-generate URL-safe slug"""
        slug = self.name.lower()
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        slug = slug.strip('-')
        self.slug = slug

product = Product(name="Tony's Chocolonely!", price=5.95)
await product.insert()
print(product.slug)  # "tony-s-chocolonely"
```

### Data Validation

```python
from beanis import Document, before_event, Insert, Update

class Product(Document):
    name: str
    price: float
    stock: int
    
    @before_event(Insert, Update)
    async def validate_business_rules(self):
        """Complex validation logic"""
        if self.price < 0:
            raise ValueError("Price must be positive")
        
        if self.stock < 0:
            raise ValueError("Stock cannot be negative")
        
        if self.price > 10000 and self.stock > 1000:
            raise ValueError("High-value + high-stock needs approval")
        
        # Normalize name
        self.name = self.name.strip().title()

product = Product(name="  laptop  ", price=999, stock=50)
await product.insert()
print(product.name)  # "Laptop"
```

### Audit Trail

```python
from datetime import datetime
from typing import Optional

class AuditLog(Document):
    entity_type: str
    entity_id: str
    action: str
    timestamp: datetime
    data: dict
    
    class Settings:
        name = "audit_logs"

class Product(Document):
    name: str
    price: float
    
    @after_event(Insert)
    async def log_insert(self):
        await AuditLog(
            entity_type="Product",
            entity_id=self.id,
            action="INSERT",
            timestamp=datetime.now(),
            data={"name": self.name, "price": self.price}
        ).insert()
    
    @after_event(Update)
    async def log_update(self):
        await AuditLog(
            entity_type="Product",
            entity_id=self.id,
            action="UPDATE",
            timestamp=datetime.now(),
            data={"name": self.name, "price": self.price}
        ).insert()
    
    @after_event(Delete)
    async def log_delete(self):
        await AuditLog(
            entity_type="Product",
            entity_id=self.id,
            action="DELETE",
            timestamp=datetime.now(),
            data={}
        ).insert()

# All operations are logged
product = Product(name="Test", price=9.99)
await product.insert()  # Creates audit log

await product.update(price=12.99)  # Creates audit log

await product.delete_self()  # Creates audit log
```

### Custom Secondary Indexes

```python
from beanis import Document, after_event, Insert, Update, Delete

class Product(Document):
    name: str
    tags: list[str]
    
    @after_event(Insert, Update)
    async def index_tags(self):
        """Maintain custom tag indexes"""
        # Add to Redis sets for each tag
        for tag in self.tags:
            await self._database.sadd(f"tag_index:{tag}", self.id)
    
    @after_event(Delete)
    async def cleanup_tag_indexes(self):
        """Remove from tag indexes"""
        for tag in self.tags:
            await self._database.srem(f"tag_index:{tag}", self.id)

# Tags are automatically indexed
product = Product(name="Laptop", tags=["electronics", "computers"])
await product.insert()

# Find products by tag (using custom index)
laptop_ids = await Product._database.smembers("tag_index:electronics")
laptops = await Product.get_many(list(laptop_ids))
```

### Notification System

```python
from beanis import Document, after_event, Insert, Update

class Product(Document):
    name: str
    price: float
    stock: int
    
    @after_event(Update)
    async def check_low_stock(self):
        """Send alert if stock is low"""
        if self.stock < 10:
            await self.send_alert(
                f"Low stock alert: {self.name} has {self.stock} units"
            )
    
    @after_event(Update)
    async def check_price_drop(self):
        """Notify customers of price drop"""
        # Get old price from Redis
        old_data = await self._database.hgetall(f"Product:{self.id}")
        old_price = float(old_data.get("price", self.price))
        
        if self.price < old_price * 0.9:  # 10% drop
            await self.send_alert(
                f"Price drop: {self.name} now ${self.price}"
            )
    
    async def send_alert(self, message: str):
        """Send alert via your notification system"""
        print(f"ALERT: {message}")
        # Send to email, SMS, push notification, etc.

product = await Product.get("prod_123")
await product.update(stock=5)
# Output: ALERT: Low stock alert: Laptop has 5 units

await product.update(price=499.99)  # Down from $999
# Output: ALERT: Price drop: Laptop now $499.99
```

## Hook Execution Order

When multiple hooks are present:

1. **Before hooks** run first (in definition order)
2. **Operation executes** (insert/update/delete)
3. **After hooks** run last (in definition order)

```python
class Product(Document):
    name: str
    
    @before_event(Insert)
    async def hook1(self):
        print("1. Before Insert - First")
    
    @before_event(Insert)
    async def hook2(self):
        print("2. Before Insert - Second")
    
    @after_event(Insert)
    async def hook3(self):
        print("4. After Insert - First")
    
    @after_event(Insert)
    async def hook4(self):
        print("5. After Insert - Second")

await product.insert()
# Output:
# 1. Before Insert - First
# 2. Before Insert - Second
# 3. [INSERT OPERATION]
# 4. After Insert - First
# 5. After Insert - Second
```

## Error Handling

If a **before hook** raises an exception, the operation is **aborted**:

```python
class Product(Document):
    price: float
    
    @before_event(Insert)
    async def validate_price(self):
        if self.price < 0:
            raise ValueError("Invalid price")

try:
    product = Product(price=-5.0)
    await product.insert()
except ValueError as e:
    print(f"Insert failed: {e}")
    # Document was NOT inserted
```

If an **after hook** raises an exception, the operation **already completed**:

```python
class Product(Document):
    name: str
    
    @after_event(Insert)
    async def send_notification(self):
        raise Exception("Notification failed")

try:
    product = Product(name="Test")
    await product.insert()
except Exception as e:
    print(f"After hook failed: {e}")
    # But document WAS inserted successfully
```

## Performance Considerations

1. **Keep hooks fast** - They run synchronously with operations
2. **Use after hooks for slow tasks** - Don't block the operation
3. **Consider background tasks** - For heavy processing
4. **Avoid circular dependencies** - Hook calling save() on same document

## Best Practices

1. **Use before hooks for validation** - Prevent bad data from reaching Redis
2. **Use after hooks for side effects** - Notifications, logging, etc.
3. **Keep hooks simple** - One clear responsibility per hook
4. **Handle errors gracefully** - Especially in after hooks
5. **Document hook behavior** - Make it clear what hooks do
6. **Test hooks separately** - Unit test hook logic

## Next Steps

- [Insert Operations](insert.md) - Using hooks with inserts
- [Update Operations](update.md) - Using hooks with updates
- [Delete Operations](delete.md) - Using hooks with deletes
- [Custom Encoders](../CUSTOM_ENCODERS.md) - Transform data in hooks
