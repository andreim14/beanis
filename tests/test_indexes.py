"""
Tests for secondary indexing using Redis Sets and Sorted Sets
"""
import pytest
from typing import Optional
from pydantic import BaseModel
from typing_extensions import Annotated

from beanis import Document, init_beanis, IndexedField


class Category(BaseModel):
    name: str
    description: str


class Product(Document):
    name: str
    description: Optional[str] = None
    # Indexed fields
    category: Annotated[str, IndexedField()]  # Set index for exact match
    price: Annotated[float, IndexedField()]   # Sorted Set for range queries
    stock: Annotated[int, IndexedField()]     # Sorted Set for range queries
    brand: Annotated[Optional[str], IndexedField()] = None

    class Settings:
        key_prefix = "Product"


@pytest.mark.asyncio
async def test_indexed_field_insert(redis_client):
    """Test that indexed fields are properly indexed on insert"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(
        id="prod-1",
        name="Laptop",
        category="electronics",
        price=999.99,
        stock=50
    )
    await product.insert()

    # Check that Set index exists
    members = await redis_client.smembers("idx:Product:category:electronics")
    assert b"prod-1" in members or "prod-1" in members

    # Check that Sorted Set index exists
    score = await redis_client.zscore("idx:Product:price", "prod-1")
    assert score == 999.99

    stock_score = await redis_client.zscore("idx:Product:stock", "prod-1")
    assert stock_score == 50


@pytest.mark.asyncio
async def test_find_by_exact_match(redis_client):
    """Test finding documents by exact match on indexed field"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert test products
    products = [
        Product(name="Laptop", category="electronics", price=999, stock=10),
        Product(name="Phone", category="electronics", price=599, stock=20),
        Product(name="Desk", category="furniture", price=299, stock=5),
        Product(name="Chair", category="furniture", price=149, stock=15),
    ]

    for p in products:
        await p.insert()

    # Find by category
    electronics = await Product.find(category="electronics")
    assert len(electronics) == 2
    assert all(p.category == "electronics" for p in electronics)

    furniture = await Product.find(category="furniture")
    assert len(furniture) == 2
    assert all(p.category == "furniture" for p in furniture)


@pytest.mark.asyncio
async def test_find_by_range_gte(redis_client):
    """Test finding documents by numeric range (>=)"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert test products with varying prices
    products = [
        Product(name="Budget Phone", category="electronics", price=199, stock=10),
        Product(name="Mid Phone", category="electronics", price=499, stock=10),
        Product(name="Premium Phone", category="electronics", price=999, stock=10),
        Product(name="Ultra Phone", category="electronics", price=1299, stock=10),
    ]

    for p in products:
        await p.insert()

    # Find products with price >= 500
    expensive = await Product.find(price__gte=500)
    assert len(expensive) == 2
    assert all(p.price >= 500 for p in expensive)


@pytest.mark.asyncio
async def test_find_by_range_lte(redis_client):
    """Test finding documents by numeric range (<=)"""
    await init_beanis(database=redis_client, document_models=[Product])

    products = [
        Product(name="Item 1", category="test", price=100, stock=10),
        Product(name="Item 2", category="test", price=250, stock=10),
        Product(name="Item 3", category="test", price=500, stock=10),
        Product(name="Item 4", category="test", price=750, stock=10),
    ]

    for p in products:
        await p.insert()

    # Find products with price <= 300
    affordable = await Product.find(price__lte=300)
    assert len(affordable) == 2
    assert all(p.price <= 300 for p in affordable)


@pytest.mark.asyncio
async def test_find_by_range_between(redis_client):
    """Test finding documents in a range"""
    await init_beanis(database=redis_client, document_models=[Product])

    products = [
        Product(name=f"Product {i}", category="test", price=float(i * 100), stock=i)
        for i in range(1, 11)
    ]

    for p in products:
        await p.insert()

    # Find products with price between 300 and 700
    mid_range = await Product.find(price__gte=300, price__lte=700)
    assert len(mid_range) == 5  # 300, 400, 500, 600, 700
    assert all(300 <= p.price <= 700 for p in mid_range)


@pytest.mark.asyncio
async def test_find_combined_filters(redis_client):
    """Test finding with multiple filters (AND logic)"""
    await init_beanis(database=redis_client, document_models=[Product])

    products = [
        Product(name="Cheap Electronics", category="electronics", price=100, stock=50),
        Product(name="Expensive Electronics", category="electronics", price=1000, stock=5),
        Product(name="Cheap Furniture", category="furniture", price=150, stock=30),
        Product(name="Expensive Furniture", category="furniture", price=900, stock=10),
    ]

    for p in products:
        await p.insert()

    # Find electronics with price >= 500
    expensive_electronics = await Product.find(
        category="electronics",
        price__gte=500
    )
    assert len(expensive_electronics) == 1
    assert expensive_electronics[0].name == "Expensive Electronics"

    # Find products with price <= 200 and stock >= 40
    cheap_high_stock = await Product.find(
        price__lte=200,
        stock__gte=40
    )
    assert len(cheap_high_stock) == 1
    assert cheap_high_stock[0].name == "Cheap Electronics"


@pytest.mark.asyncio
async def test_index_update_on_delete(redis_client):
    """Test that indexes are cleaned up when document is deleted"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(
        id="delete-test",
        name="Test Product",
        category="test",
        price=100,
        stock=10
    )
    await product.insert()

    # Verify indexes exist
    members = await redis_client.smembers("idx:Product:category:test")
    assert b"delete-test" in members or "delete-test" in members

    # Delete product
    await product.delete_self()

    # Verify indexes are cleaned up
    members_after = await redis_client.smembers("idx:Product:category:test")
    assert b"delete-test" not in members_after and "delete-test" not in members_after

    score = await redis_client.zscore("idx:Product:price", "delete-test")
    assert score is None


@pytest.mark.asyncio
async def test_find_no_results(redis_client):
    """Test finding with filters that match nothing"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(
        name="Only Product",
        category="electronics",
        price=500,
        stock=10
    )
    await product.insert()

    # Find with non-matching filter
    results = await Product.find(category="furniture")
    assert len(results) == 0

    # Find with out-of-range filter
    results = await Product.find(price__gte=1000)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_optional_indexed_field(redis_client):
    """Test indexed field that is optional (can be None)"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Product without brand
    p1 = Product(
        name="Generic Product",
        category="test",
        price=100,
        stock=10,
        brand=None
    )
    await p1.insert()

    # Product with brand
    p2 = Product(
        name="Brand Product",
        category="test",
        price=200,
        stock=20,
        brand="Apple"
    )
    await p2.insert()

    # Find by brand
    branded = await Product.find(brand="Apple")
    assert len(branded) == 1
    assert branded[0].name == "Brand Product"


@pytest.mark.asyncio
async def test_find_without_filters_returns_all(redis_client):
    """Test that find() without filters returns all documents"""
    await init_beanis(database=redis_client, document_models=[Product])

    products = [
        Product(name=f"Product {i}", category="test", price=float(i * 100), stock=i)
        for i in range(1, 6)
    ]

    for p in products:
        await p.insert()

    # Find without filters
    all_products = await Product.find()
    assert len(all_products) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
