# Indexes

Beanis uses Redis data structures (Sorted Sets and Sets) to enable fast queries on your documents.

## Overview

Unlike MongoDB which has a flexible indexing system, Redis indexing in Beanis is simpler but powerful:

- **Sorted Sets** - For numeric fields (support range queries)
- **Sets** - For string/categorical fields (exact match only)
- **Geo Indexes** - For geo-spatial fields (proximity queries)
- **Hash** - Document storage (no indexing)

## Defining Indexes

Use the `Indexed()` type annotation to mark fields for indexing:

```python
from beanis import Document, Indexed, GeoPoint

class Product(Document):
    name: str                    # Not indexed - no queries
    price: Indexed[float]        # Indexed - range queries
    category: Indexed[str]       # Indexed - exact match
    stock: int                   # Not indexed - no queries
    location: Indexed[GeoPoint]  # Indexed - geo queries

    class Settings:
        name = "products"
```

## Numeric Indexes (Sorted Sets)

Numeric fields (`int`, `float`) are stored in Redis Sorted Sets:

```python
class Product(Document):
    name: str
    price: Indexed(float)  # Stored in sorted set
    stock: Indexed(int)    # Stored in sorted set

# Enables range queries
expensive = await Product.find(Product.price > 100).to_list()
affordable = await Product.find(
    Product.price >= 10,
    Product.price <= 50
).to_list()

# Stock queries
low_stock = await Product.find(Product.stock < 10).to_list()
```

**Redis Structure**:
```
Key: idx:Product:price
Type: Sorted Set
Content: {document_id: price_value, ...}

Example:
idx:Product:price -> {
    "prod_1": 9.99,
    "prod_2": 15.50,
    "prod_3": 25.00
}
```

**Supported Operations**:
- `==` - Exact match
- `>`, `>=` - Greater than
- `<`, `<=` - Less than
- Range combinations

## Categorical Indexes (Sets)

String fields are stored in Redis Sets (one set per unique value):

```python
class Product(Document):
    name: str
    category: Indexed(str)  # Stored in sets
    brand: Indexed(str)     # Stored in sets

# Enables exact match queries
books = await Product.find(Product.category == "books").to_list()
nike = await Product.find(Product.brand == "Nike").to_list()
```

**Redis Structure**:
```
Key: idx:Product:category:books
Type: Set
Content: {document_id, document_id, ...}

Example:
idx:Product:category:books -> {"prod_1", "prod_2", "prod_5"}
idx:Product:category:electronics -> {"prod_3", "prod_4"}
```

**Supported Operations**:
- `==` - Exact match only

**Not Supported**:
- Partial matches (use full-text search module)
- Case-insensitive search
- Wildcards

## Index Creation

Indexes are created automatically when you insert documents:

```python
# Define indexed fields
class Product(Document):
    price: Indexed(float)
    category: Indexed(str)

# Initialize
await init_beanis(database=redis_client, document_models=[Product])

# Insert - indexes created automatically
product = Product(
    name="Laptop",
    price=999.99,
    category="electronics"
)
await product.insert()

# This creates:
# 1. Hash: Product:{id}
# 2. Sorted Set entry: idx:Product:price
# 3. Set entry: idx:Product:category:electronics
```

## Index Updates

Indexes are automatically updated when documents change:

```python
product = await Product.get("product_id_123")

# Update price - sorted set automatically updated
await product.update(price=799.99)

# Update category - sets automatically updated
await product.update(category="computers")
# Old set entry removed, new set entry added
```

## Multiple Indexes

You can index multiple fields on the same document:

```python
class Product(Document):
    name: str
    price: Indexed(float)
    stock: Indexed(int)
    category: Indexed(str)
    brand: Indexed(str)

# Each indexed field has its own Redis structure
# - idx:Product:price (sorted set)
# - idx:Product:stock (sorted set)
# - idx:Product:category:{value} (sets)
# - idx:Product:brand:{value} (sets)
```

## Query Performance

### Indexed Queries (Fast)

```python
# These queries use indexes - O(log N) or O(1)
products = await Product.find(Product.price < 100).to_list()
books = await Product.find(Product.category == "books").to_list()
```

### Non-Indexed Queries (Slow)

```python
# This field is NOT indexed - requires full scan
class Product(Document):
    name: str  # Not indexed!
    price: Indexed(float)

# This will NOT work (name not indexed):
# products = await Product.find(Product.name == "Laptop").to_list()

# You must use get_all() and filter manually:
all_products = await Product.all()
laptops = [p for p in all_products if p.name == "Laptop"]
```

**Rule**: Only `Indexed()` fields support `.find()` queries!

## Index Storage Cost

Each index uses Redis memory:

```python
class Product(Document):
    price: Indexed(float)     # ~50 bytes per document
    category: Indexed(str)    # ~40 bytes per document per unique value

# For 10,000 products with 5 categories:
# - price index: ~500 KB
# - category index: ~200 KB (5 sets)
# - documents: ~5-10 MB (depends on size)
```

**Best Practice**: Only index fields you'll query frequently.

## Compound Queries

Beanis supports querying multiple indexed fields:

```python
# Both fields must be indexed
products = await Product.find(
    Product.price >= 10,
    Product.price <= 50,
    Product.category == "electronics"
).to_list()
```

**How it works**:
1. Query price index (sorted set range)
2. Query category index (set members)
3. Intersect results (in Python)

## Index Limitations

### No Full-Text Search

```python
# Not supported - no partial matching
# products = await Product.find(Product.name.contains("Lap")).to_list()

# Use Redis RediSearch module for full-text search
# Or filter in memory:
all_products = await Product.all()
matches = [p for p in all_products if "Lap" in p.name]
```

### No Case-Insensitive Search

```python
# Case-sensitive only
books = await Product.find(Product.category == "Books").to_list()  # Won't find "books"

# Workaround: Store lowercase
class Product(Document):
    category_lower: Indexed(str)
    
    @before_event(Insert, Update)
    async def normalize_category(self):
        self.category_lower = self.category.lower()

# Query lowercase
books = await Product.find(Product.category_lower == "books").to_list()
```

### No Date Range Queries

```python
from datetime import datetime

class Product(Document):
    created_at: datetime  # Can't index datetime directly

# Workaround: Store as timestamp (numeric)
class Product(Document):
    created_at: datetime
    created_timestamp: Indexed(float)
    
    @before_event(Insert)
    async def set_timestamp(self):
        self.created_timestamp = self.created_at.timestamp()

# Query by timestamp
recent = await Product.find(
    Product.created_timestamp > datetime(2024, 1, 1).timestamp()
).to_list()
```

## Index Maintenance

### Automatic Cleanup

Indexes are cleaned up automatically on:
- Document deletion
- Field updates
- Document updates

```python
# Delete - removes from all indexes
await product.delete_self()

# Update category - removes from old set, adds to new set
await product.update(category="new_category")
```

### Manual Index Check

Check what's in an index:

```python
# Direct Redis access (for debugging)
redis_client = Product._database

# Check sorted set index
price_index = await redis_client.zrange("idx:Product:price", 0, -1, withscores=True)
print(price_index)  # [(doc_id, score), ...]

# Check set index
books_set = await redis_client.smembers("idx:Product:category:books")
print(books_set)  # {doc_id, doc_id, ...}
```

## Advanced Patterns

### Geo-Spatial Indexing

Redis supports geo-spatial queries, and Beanis provides built-in support through the GeoPoint type.

```python
from beanis import Document, Indexed, GeoPoint
from beanis.odm.indexes import IndexManager

class Store(Document):
    name: str
    location: Indexed[GeoPoint]  # Geo index

    class Settings:
        name = "stores"

# Create a store
store = Store(
    name="Downtown Store",
    location=GeoPoint(longitude=-122.4, latitude=37.8)
)
await store.insert()

# Find nearby stores (within 10km)
from beanis.odm.indexes import IndexManager

nearby_ids = await IndexManager.find_by_geo_radius(
    redis_client=Store._database,
    document_class=Store,
    field_name="location",
    longitude=-122.4,
    latitude=37.8,
    radius=10,
    unit="km"
)

# Get the actual documents
nearby_stores = await Store.get_many(nearby_ids)

# Find nearby stores with distances
nearby_with_dist = await IndexManager.find_by_geo_radius_with_distance(
    redis_client=Store._database,
    document_class=Store,
    field_name="location",
    longitude=-122.4,
    latitude=37.8,
    radius=10,
    unit="km"
)

for store_id, distance in nearby_with_dist:
    print(f"Store {store_id} is {distance} km away")
```

**Supported units**: `m` (meters), `km` (kilometers), `mi` (miles), `ft` (feet)

**How it works**:
- Uses Redis GEOADD command to store locations
- Uses Redis GEORADIUS for proximity queries
- Automatically maintains geo index on insert/update/delete

### Multi-Value Indexing

For list fields, consider storing as separate documents or using a workaround:

```python
# Not directly supported
class Product(Document):
    tags: List[str]  # Can't index list

# Workaround 1: Store each tag as indexed field (if few tags)
class Product(Document):
    tag1: Indexed(str) = ""
    tag2: Indexed(str) = ""
    tag3: Indexed(str) = ""

# Workaround 2: Manual secondary index
class Product(Document):
    tags: List[str]
    
    async def index_tags(self):
        """Manually add to tag sets"""
        for tag in self.tags:
            await self._database.sadd(f"tags:{tag}", self.id)
```

## Best Practices

1. **Index selectively** - Only fields you'll query frequently
2. **Use correct types** - Numeric for ranges, strings for exact match
3. **Monitor memory** - Indexes consume Redis memory
4. **Lowercase strings** - For case-insensitive queries
5. **Timestamps for dates** - Convert datetime to float for range queries
6. **Test performance** - Profile with realistic data volumes

## Comparison with MongoDB

| Feature | MongoDB | Beanis (Redis) |
|---------|---------|----------------|
| Numeric range | ✅ Compound indexes | ✅ Sorted sets |
| Exact match | ✅ Single field | ✅ Sets |
| Full-text search | ✅ Text indexes | ❌ Need RediSearch |
| Compound indexes | ✅ Multi-field | ⚠️ Manual intersection |
| Geo-spatial | ✅ 2dsphere | ✅ Built-in (GEORADIUS) |
| Index creation | Manual | ✅ Automatic |
| Array indexing | ✅ Multikey | ❌ Not supported |

## Next Steps

- [Find Operations](find.md) - Query using indexes
- [Event Hooks](actions.md) - Maintain custom indexes
- [Custom Encoders](../CUSTOM_ENCODERS.md) - Store complex types
