"""
Core tests for Beanis Redis ODM
Tests basic CRUD operations, TTL, batch operations, etc.
"""
import pytest
from typing import Optional
from pydantic import BaseModel

from beanis import Document, init_beanis


class Category(BaseModel):
    name: str
    description: str


class Product(Document):
    name: str
    description: Optional[str] = None
    price: float
    category: Optional[Category] = None
    stock: int = 0

    class Settings:
        key_prefix = "Product"


@pytest.mark.asyncio
async def test_insert_and_get(redis_client):
    """Test basic insert and get operations"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Create and insert a product
    product = Product(
        id="test-1",
        name="Test Product",
        price=99.99,
        stock=10
    )
    await product.insert()

    # Retrieve it
    found = await Product.get("test-1")
    assert found is not None
    assert found.name == "Test Product"
    assert found.price == 99.99
    assert found.stock == 10


@pytest.mark.asyncio
async def test_get_nonexistent(redis_client):
    """Test getting a document that doesn't exist"""
    await init_beanis(database=redis_client, document_models=[Product])

    found = await Product.get("nonexistent")
    assert found is None


@pytest.mark.asyncio
async def test_exists(redis_client):
    """Test exists check"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="test-exists", name="Test", price=10.0)
    await product.insert()

    assert await Product.exists("test-exists") is True
    assert await Product.exists("not-exists") is False


@pytest.mark.asyncio
async def test_update_fields(redis_client):
    """Test updating specific fields"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="test-update", name="Original", price=50.0, stock=5)
    await product.insert()

    # Update fields
    await product.update(price=75.0, stock=10)

    # Verify updates
    found = await Product.get("test-update")
    assert found.price == 75.0
    assert found.stock == 10
    assert found.name == "Original"  # Should be unchanged


@pytest.mark.asyncio
async def test_increment_field(redis_client):
    """Test incrementing a numeric field"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="test-incr", name="Test", price=100.0, stock=5)
    await product.insert()

    # Increment stock
    new_stock = await product.increment_field("stock", 3)
    assert new_stock == 8

    # Increment price (float)
    new_price = await product.increment_field("price", 25.5)
    assert new_price == 125.5

    # Verify in database
    found = await Product.get("test-incr")
    assert found.stock == 8
    assert found.price == 125.5


@pytest.mark.asyncio
async def test_delete(redis_client):
    """Test deleting a document"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="test-delete", name="Test", price=10.0)
    await product.insert()

    assert await Product.exists("test-delete") is True

    await product.delete_self()

    assert await Product.exists("test-delete") is False


@pytest.mark.asyncio
async def test_ttl(redis_client):
    """Test TTL operations"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert with TTL
    product = Product(id="test-ttl", name="Test", price=10.0)
    await product.insert(ttl=3600)

    # Check TTL
    ttl = await product.get_ttl()
    assert ttl is not None
    assert ttl > 0
    assert ttl <= 3600

    # Set new TTL
    await product.set_ttl(7200)
    new_ttl = await product.get_ttl()
    assert new_ttl > 3600

    # Remove TTL
    await product.persist()
    persistent_ttl = await product.get_ttl()
    assert persistent_ttl == -1  # -1 means no TTL


@pytest.mark.asyncio
async def test_count_and_all(redis_client):
    """Test counting and getting all documents"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert multiple products
    products = [
        Product(name=f"Product {i}", price=float(i * 10), stock=i)
        for i in range(1, 6)
    ]

    for product in products:
        await product.insert()

    # Test count
    count = await Product.count()
    assert count == 5

    # Test get all
    all_products = await Product.all()
    assert len(all_products) == 5

    # Test pagination
    first_two = await Product.all(limit=2)
    assert len(first_two) == 2

    next_two = await Product.all(skip=2, limit=2)
    assert len(next_two) == 2


@pytest.mark.asyncio
async def test_insert_many(redis_client):
    """Test bulk insert operation"""
    await init_beanis(database=redis_client, document_models=[Product])

    products = [
        Product(name=f"Bulk Product {i}", price=float(i * 5), stock=i * 2)
        for i in range(1, 11)
    ]

    # Bulk insert
    inserted = await Product.insert_many(products)
    assert len(inserted) == 10

    # Verify all were inserted
    count = await Product.count()
    assert count == 10


@pytest.mark.asyncio
async def test_get_many(redis_client):
    """Test bulk get operation"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert products with specific IDs
    ids = ["prod-1", "prod-2", "prod-3"]
    products = [
        Product(id=id, name=f"Product {id}", price=10.0)
        for id in ids
    ]

    for product in products:
        await product.insert()

    # Get many
    found = await Product.get_many(ids)
    assert len(found) == 3
    assert all(p is not None for p in found)

    # Test with mix of existing and non-existing
    mixed_ids = ["prod-1", "nonexistent", "prod-3"]
    mixed_found = await Product.get_many(mixed_ids)
    assert len(mixed_found) == 3
    assert mixed_found[0] is not None
    assert mixed_found[1] is None
    assert mixed_found[2] is not None


@pytest.mark.asyncio
async def test_delete_many(redis_client):
    """Test bulk delete operation"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert products
    ids = [f"del-{i}" for i in range(1, 6)]
    products = [
        Product(id=id, name=f"Product {id}", price=10.0)
        for id in ids
    ]

    for product in products:
        await product.insert()

    # Delete some
    deleted = await Product.delete_many(["del-1", "del-2", "del-3"])
    assert deleted == 3

    # Verify
    assert await Product.count() == 2
    assert await Product.exists("del-1") is False
    assert await Product.exists("del-4") is True


@pytest.mark.asyncio
async def test_delete_all(redis_client):
    """Test deleting all documents"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert multiple products
    for i in range(1, 6):
        product = Product(name=f"Product {i}", price=float(i * 10))
        await product.insert()

    assert await Product.count() == 5

    # Delete all
    deleted = await Product.delete_all()
    assert deleted == 5

    assert await Product.count() == 0


@pytest.mark.asyncio
async def test_nested_model(redis_client):
    """Test storing nested Pydantic models"""
    await init_beanis(database=redis_client, document_models=[Product])

    category = Category(name="Electronics", description="Electronic items")
    product = Product(
        id="nested-test",
        name="Laptop",
        price=999.99,
        category=category
    )

    await product.insert()

    # Retrieve and verify nested model
    found = await Product.get("nested-test")
    assert found is not None
    assert found.category is not None
    assert found.category.name == "Electronics"
    assert found.category.description == "Electronic items"


@pytest.mark.asyncio
async def test_auto_id_generation(redis_client):
    """Test automatic ID generation"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(name="No ID Product", price=50.0)
    await product.insert()

    # ID should be auto-generated
    assert product.id is not None
    assert len(product.id) > 0

    # Should be retrievable
    found = await Product.get(product.id)
    assert found is not None
    assert found.name == "No ID Product"


@pytest.mark.asyncio
async def test_field_operations(redis_client):
    """Test individual field get/set operations"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="field-test", name="Test", price=100.0, stock=10)
    await product.insert()

    # Get individual field
    price = await product.get_field("price")
    assert price == "100.0" or price == 100.0  # May be string from Redis

    # Set individual field
    await product.set_field("stock", 20)
    updated_stock = await product.get_field("stock")
    assert updated_stock == "20" or updated_stock == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
