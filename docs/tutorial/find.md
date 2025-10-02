# Find Documents

Beanis provides a powerful query interface for finding documents in Redis.

## Find One

Get a single document by ID:

```python
product = await Product.get("product_id_123")
if product:
    print(product.name)
```

## Find with Conditions

Query documents using indexed fields:

```python
from beanis import Document, Indexed

class Product(Document):
    name: str
    price: Indexed(float)  # Must be indexed for queries
    category: Indexed(str)
    
    class Settings:
        name = "products"

# Range query on numeric field
products = await Product.find(
    Product.price >= 10.0,
    Product.price <= 50.0
).to_list()

# Exact match on categorical field  
electronics = await Product.find(
    Product.category == "electronics"
).to_list()
```

## Query Methods

### to_list()

Get all matching documents as a list:

```python
products = await Product.find(Product.price < 100).to_list()
```

### first_or_none()

Get the first matching document or None:

```python
product = await Product.find(Product.price > 1000).first_or_none()
if product:
    print(product.name)
```

### limit()

Limit the number of results:

```python
# Get first 10 products
products = await Product.find(Product.price > 10).limit(10).to_list()
```

## Get All Documents

Retrieve all documents (uses sorted set tracking):

```python
# Get all products
all_products = await Product.all()

# With pagination
page1 = await Product.all(limit=10)
page2 = await Product.all(skip=10, limit=10)

# Sort descending (newest first)
recent = await Product.all(sort_desc=True, limit=5)
```

## Count Documents

Count total documents:

```python
count = await Product.count()
print(f"Total products: {count}")
```

## Check Existence

Check if a document exists:

```python
exists = await Product.exists("product_id_123")
if exists:
    print("Product exists")
```

## Batch Get

Get multiple documents by their IDs:

```python
ids = ["id1", "id2", "id3"]
products = await Product.get_many(ids)

for product in products:
    if product:  # Some IDs might not exist
        print(product.name)
```

## Important Notes

1. **Indexed fields required for queries** - Only indexed fields support `find()` queries
2. **Exact match vs range queries**:
   - Numeric fields (int, float): Support range queries (>, <, >=, <=, ==)
   - String fields: Support exact match only (==)
3. **all() uses document tracking** - Returns documents in insertion order

## Query Examples

### Find by price range:
```python
affordable = await Product.find(
    Product.price >= 5.0,
    Product.price <= 20.0
).to_list()
```

### Find by category:
```python
books = await Product.find(Product.category == "books").to_list()
```

### Find with limit:
```python
top_10 = await Product.find(Product.price > 0).limit(10).to_list()
```

### Find first match:
```python
expensive = await Product.find(Product.price > 1000).first_or_none()
```

## Next Steps

- [Update Operations](update.md) - Modify documents
- [Delete Operations](delete.md) - Remove documents
- [Indexes](indexes.md) - Learn about indexing
