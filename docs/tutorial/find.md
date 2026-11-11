# Find Documents

Beanis finds documents by ID or by keyword filters on indexed fields.

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

# Range query on numeric field (bounds are inclusive)
products = await Product.find(price__gte=10.0, price__lte=50.0)

# Exact match on categorical field
electronics = await Product.find(category="electronics")

# Several filters are ANDed
cheap_books = await Product.find(category="books", price__lt=20.0)
```

`find()` takes keyword filters and returns a `List[Product]` directly; there is
no query builder to chain or `.to_list()` to call. Supported suffixes on numeric
indexed fields are `__gte`, `__lte`, `__gt` and `__lt`. Calling `find()` with no
filters returns every document, like `all()`.

## First Match, Sorting and Limits

`find()` has no `first_or_none()`, `sort()` or `limit()`. Results come back in
no particular order, so sort and slice in Python:

```python
products = await Product.find(price__gt=10)

# First match, or None
product = products[0] if products else None

# Ten cheapest
cheapest = sorted(products, key=lambda p: p.price)[:10]
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

1. **Indexed fields required for queries** - `find()` only accepts indexed
   fields; filtering on any other field raises `ValueError`
2. **Exact match vs range queries**:
   - Numeric fields (int, float): ranges only (`__gte`, `__lte`, `__gt`,
     `__lt`); for an exact value use `price__gte=10, price__lte=10`
   - String fields: exact match only (`category="books"`)
3. **all() uses document tracking** - Returns documents in insertion order

## Query Examples

### Find by price range:
```python
affordable = await Product.find(price__gte=5.0, price__lte=20.0)
```

### Find by category:
```python
books = await Product.find(category="books")
```

### Find most expensive 10:
```python
top_10 = sorted(await Product.find(price__gt=0), key=lambda p: p.price, reverse=True)[:10]
```

### Find first match:
```python
matches = await Product.find(price__gt=1000)
expensive = matches[0] if matches else None
```

## Next Steps

- [Update Operations](update.md) - Modify documents
- [Delete Operations](delete.md) - Remove documents
- [Indexes](indexes.md) - Learn about indexing
