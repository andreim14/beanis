# Delete Documents

Beanis provides several methods for deleting documents from Redis.

## Delete Single Document

### Delete by Instance

Use `delete_self()` to delete a document instance:

```python
from beanis import Document

class Product(Document):
    name: str
    price: float
    
    class Settings:
        name = "products"

# Get and delete
product = await Product.get("product_id_123")
await product.delete_self()

# Verify deletion
exists = await Product.exists("product_id_123")
print(exists)  # False
```

### Delete by ID

Use the class method `delete()` to delete by ID without fetching:

```python
# Delete without fetching
await Product.delete("product_id_123")

# More efficient than:
# product = await Product.get("product_id_123")
# await product.delete_self()
```

## Delete Multiple Documents

Use `delete_many()` to delete multiple documents by their IDs:

```python
# Delete multiple products
ids = ["id1", "id2", "id3", "id4", "id5"]
await Product.delete_many(ids)

# Verify deletions
for id in ids:
    exists = await Product.exists(id)
    print(f"{id} exists: {exists}")  # All False
```

This uses Redis pipelines for efficient batch deletion.

## Delete All Documents

Use `delete_all()` to remove all documents of a type:

```python
# Delete all products
await Product.delete_all()

# Verify
count = await Product.count()
print(count)  # 0
```

**Warning:** This deletes ALL documents and cannot be undone!

## Check Before Delete

Always check if a document exists before attempting deletion:

```python
# Safe deletion
if await Product.exists("product_id_123"):
    await Product.delete("product_id_123")
    print("Deleted")
else:
    print("Product not found")
```

## Delete with Event Hooks

Run custom logic before/after deletions:

```python
from beanis import before_event, after_event, Delete
from datetime import datetime

class Product(Document):
    name: str
    price: float
    
    @before_event(Delete)
    async def backup_before_delete(self):
        """Log deletion for audit trail"""
        print(f"Deleting product: {self.name} at {datetime.now()}")
    
    @after_event(Delete)
    async def cleanup_after_delete(self):
        """Clean up related resources"""
        print(f"Product {self.id} deleted successfully")

product = await Product.get("product_id_123")
await product.delete_self()
# Output: Deleting product: Test Product at 2025-01-15 10:30:00
#         Product product_id_123 deleted successfully
```

## What Gets Deleted

When you delete a document, Beanis removes:

1. **The document Hash** - `Product:{id}`
2. **Index entries** - All sorted set/set entries for indexed fields
3. **Tracking entry** - Entry in `all:Product` sorted set

```python
# For this document:
class Product(Document):
    name: str
    price: Indexed(float)
    category: Indexed(str)

# Deletion removes:
# 1. Hash: Product:{id}
# 2. Index: idx:Product:price (sorted set entry)
# 3. Index: idx:Product:category:{value} (set entry)
# 4. Tracking: all:Product (sorted set entry)
```

## Bulk Delete Pattern

Efficient pattern for conditional bulk deletion:

```python
# Delete all products below $5
cheap_products = await Product.find(Product.price < 5.0).to_list()
ids_to_delete = [p.id for p in cheap_products]

if ids_to_delete:
    await Product.delete_many(ids_to_delete)
    print(f"Deleted {len(ids_to_delete)} products")
```

## Delete with TTL Alternative

Instead of deleting, consider using TTL for automatic expiration:

```python
# Instead of deleting immediately
await product.delete_self()

# Consider expiring after 30 days
await product.set_ttl(30 * 24 * 3600)  # 30 days

# Document auto-deletes after TTL expires
```

This is useful for soft-deletes and compliance requirements.

## Delete vs Expire

**Delete (Immediate)**:
```python
await product.delete_self()  # Gone immediately
```

**Expire (Scheduled)**:
```python
await product.set_ttl(3600)  # Gone in 1 hour
```

Choose based on your use case:
- **Delete**: When you need immediate removal
- **Expire**: For automatic cleanup, soft deletes, caching

## Transaction-Safe Deletion

For critical operations, verify deletion succeeded:

```python
async def safe_delete(product_id: str) -> bool:
    """Delete and verify"""
    if not await Product.exists(product_id):
        return False
    
    await Product.delete(product_id)
    
    # Verify deletion
    still_exists = await Product.exists(product_id)
    return not still_exists

success = await safe_delete("product_id_123")
if success:
    print("Successfully deleted")
else:
    print("Deletion failed")
```

## Cascade Deletion Pattern

Beanis doesn't support automatic cascade deletes, but you can implement them:

```python
class Order(Document):
    product_id: str
    quantity: int
    
    class Settings:
        name = "orders"

async def delete_product_cascade(product_id: str):
    """Delete product and all related orders"""
    # Find all related orders
    # Note: This requires get_all and filter in memory
    # since we can't query on non-indexed product_id
    all_orders = await Order.all()
    related_orders = [o for o in all_orders if o.product_id == product_id]
    
    # Delete related orders
    if related_orders:
        order_ids = [o.id for o in related_orders]
        await Order.delete_many(order_ids)
    
    # Delete product
    await Product.delete(product_id)
    
    print(f"Deleted product and {len(related_orders)} related orders")

await delete_product_cascade("product_id_123")
```

## Delete Performance

Deletion performance by method:

- `delete(id)` - O(1) - Fast, single Redis DEL command
- `delete_self()` - O(1) - Same as delete(id)
- `delete_many(ids)` - O(N) - Uses pipeline, efficient for bulk
- `delete_all()` - O(N) - Deletes all documents, can be slow for large collections

## Important Notes

1. **Deletion is permanent** - No undo unless you have backups
2. **Indexes are cleaned up** - All index entries automatically removed
3. **Use pipelines for bulk** - `delete_many()` is optimized
4. **Consider TTL instead** - For temporary data or soft deletes
5. **Event hooks run** - Before/after delete hooks are triggered

## Examples

### Soft Delete with Flag

```python
class Product(Document):
    name: str
    price: float
    deleted: bool = False
    
    async def soft_delete(self):
        """Mark as deleted instead of removing"""
        await self.update(deleted=True)
        # Optionally set TTL for eventual removal
        await self.set_ttl(30 * 24 * 3600)  # Remove after 30 days
    
    @classmethod
    async def get_active(cls, product_id: str):
        """Get only non-deleted products"""
        product = await cls.get(product_id)
        if product and not product.deleted:
            return product
        return None

# Soft delete
product = await Product.get("prod_123")
await product.soft_delete()

# Won't find soft-deleted
active = await Product.get_active("prod_123")
print(active)  # None
```

### Batch Delete with Confirmation

```python
async def batch_delete_with_confirm(ids: list[str], model_class):
    """Delete multiple documents with confirmation"""
    # Check how many exist
    existing = []
    for id in ids:
        if await model_class.exists(id):
            existing.append(id)
    
    if not existing:
        print("No documents to delete")
        return 0
    
    print(f"Found {len(existing)} documents to delete")
    
    # Delete
    await model_class.delete_many(existing)
    
    # Verify
    deleted_count = 0
    for id in existing:
        if not await model_class.exists(id):
            deleted_count += 1
    
    print(f"Successfully deleted {deleted_count}/{len(existing)} documents")
    return deleted_count

# Use it
ids = ["id1", "id2", "id3", "id4", "id5"]
deleted = await batch_delete_with_confirm(ids, Product)
```

### Delete Old Records

```python
from datetime import datetime, timedelta

class Product(Document):
    name: str
    created_at: datetime
    
    class Settings:
        name = "products"

async def delete_old_products(days: int = 30):
    """Delete products older than N days"""
    cutoff = datetime.now() - timedelta(days=days)
    
    # Get all products (Redis doesn't support date queries without custom indexing)
    all_products = await Product.all()
    
    # Filter in memory
    old_products = [
        p for p in all_products 
        if p.created_at < cutoff
    ]
    
    if old_products:
        ids = [p.id for p in old_products]
        await Product.delete_many(ids)
        print(f"Deleted {len(ids)} old products")
    else:
        print("No old products to delete")

# Delete products older than 90 days
await delete_old_products(days=90)
```

## Next Steps

- [Indexes](indexes.md) - Learn about Redis indexing
- [Event Hooks](actions.md) - Document lifecycle events
- [Find Operations](find.md) - Query documents before deletion
