"""
Comprehensive tests for document operations to increase code coverage
Tests: save(), update(), delete_many(), all(), count(), exists()
"""
import pytest
from typing import Optional
from pydantic import BaseModel
from beanis import Document, init_beanis


class Category(BaseModel):
    name: str
    description: str


class TestProduct(Document):
    name: str
    price: float
    stock: int = 0
    category: Optional[Category] = None

    class Settings:
        name = "test_products"


@pytest.mark.asyncio
async def test_save_insert_new_document(redis_client):
    """Test save() method inserts a new document"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    product = TestProduct(name="New Product", price=9.99, stock=10)
    await product.save()

    # Verify it was inserted
    found = await TestProduct.get(product.id)
    assert found is not None
    assert found.name == "New Product"
    assert found.price == 9.99


@pytest.mark.asyncio
async def test_save_update_existing_document(redis_client):
    """Test save() method updates an existing document"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    product = TestProduct(name="Original", price=5.00, stock=10)
    await product.insert()

    # Modify and save
    product.name = "Updated"
    product.price = 7.50
    await product.save()

    # Verify update
    found = await TestProduct.get(product.id)
    assert found.name == "Updated"
    assert found.price == 7.50
    assert found.stock == 10  # Unchanged field


@pytest.mark.asyncio
async def test_update_single_field(redis_client):
    """Test update() with a single field"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    product = TestProduct(name="Test", price=10.0, stock=5)
    await product.insert()

    await product.update(price=12.5)

    found = await TestProduct.get(product.id)
    assert found.price == 12.5
    assert found.stock == 5  # Unchanged


@pytest.mark.asyncio
async def test_update_multiple_fields(redis_client):
    """Test update() with multiple fields"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    product = TestProduct(name="Test", price=10.0, stock=5)
    await product.insert()

    await product.update(price=15.0, stock=20, name="Updated")

    found = await TestProduct.get(product.id)
    assert found.price == 15.0
    assert found.stock == 20
    assert found.name == "Updated"


@pytest.mark.asyncio
async def test_update_replaces_document(redis_client):
    """Test that update() preserves all fields"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    product = TestProduct(
        name="Test",
        price=10.0,
        stock=100,
        category=Category(name="Original", description="Desc")
    )
    await product.insert()

    # Update just price - other fields should remain
    await product.update(price=25.0)

    found = await TestProduct.get(product.id)
    assert found.price == 25.0
    assert found.stock == 100  # Should be unchanged
    assert found.category.name == "Original"  # Should be unchanged


@pytest.mark.asyncio
async def test_delete_many_documents(redis_client):
    """Test delete_many() method"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    # Create multiple documents
    products = [
        TestProduct(name=f"Product {i}", price=float(i))
        for i in range(5)
    ]
    await TestProduct.insert_many(products)
    ids = [p.id for p in products]

    # Delete first 3
    await TestProduct.delete_many(ids[:3])

    # Verify deletion
    remaining = await TestProduct.all()
    assert len(remaining) == 2
    remaining_ids = [p.id for p in remaining]
    assert ids[3] in remaining_ids
    assert ids[4] in remaining_ids


@pytest.mark.asyncio
async def test_all_no_parameters(redis_client):
    """Test all() with no parameters"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    # Create 5 products
    products = [
        TestProduct(name=f"Product {i}", price=float(i))
        for i in range(5)
    ]
    await TestProduct.insert_many(products)

    # Get all
    all_products = await TestProduct.all()
    assert len(all_products) == 5


@pytest.mark.asyncio
async def test_all_with_limit(redis_client):
    """Test all() with limit parameter"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    # Create 10 products
    products = [
        TestProduct(name=f"Product {i}", price=float(i))
        for i in range(10)
    ]
    await TestProduct.insert_many(products)

    # Get first 5
    limited = await TestProduct.all(limit=5)
    assert len(limited) == 5


@pytest.mark.asyncio
async def test_all_with_skip(redis_client):
    """Test all() with skip parameter"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    # Create 10 products
    products = [
        TestProduct(name=f"Product {i}", price=float(i))
        for i in range(10)
    ]
    await TestProduct.insert_many(products)

    # Skip first 5
    skipped = await TestProduct.all(skip=5)
    assert len(skipped) == 5


@pytest.mark.asyncio
async def test_all_with_pagination(redis_client):
    """Test all() with both skip and limit (pagination)"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    # Create 10 products
    products = [
        TestProduct(name=f"Product {i}", price=float(i))
        for i in range(10)
    ]
    await TestProduct.insert_many(products)

    # Get page 2 (skip 5, take 3)
    page2 = await TestProduct.all(skip=5, limit=3)
    assert len(page2) == 3


@pytest.mark.asyncio
async def test_all_sort_descending(redis_client):
    """Test all() with sort_desc=True"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    # Create products with delays to ensure ordering
    import asyncio
    for i in range(3):
        product = TestProduct(name=f"Product {i}", price=float(i))
        await product.insert()
        await asyncio.sleep(0.01)  # Small delay

    # Get in descending order (newest first)
    desc = await TestProduct.all(sort_desc=True)
    assert len(desc) == 3
    assert desc[0].name == "Product 2"  # Newest
    assert desc[2].name == "Product 0"  # Oldest


@pytest.mark.asyncio
async def test_all_empty_collection(redis_client):
    """Test all() on empty collection"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    all_products = await TestProduct.all()
    assert all_products == []


@pytest.mark.asyncio
async def test_count_documents(redis_client):
    """Test count() method"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    # Initially 0
    count = await TestProduct.count()
    assert count == 0

    # Add 5
    products = [
        TestProduct(name=f"Product {i}", price=float(i))
        for i in range(5)
    ]
    await TestProduct.insert_many(products)

    # Should be 5
    count = await TestProduct.count()
    assert count == 5


@pytest.mark.asyncio
async def test_exists_true(redis_client):
    """Test exists() returns True for existing document"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    product = TestProduct(name="Test", price=10.0)
    await product.insert()

    exists = await TestProduct.exists(product.id)
    assert exists is True


@pytest.mark.asyncio
async def test_exists_false(redis_client):
    """Test exists() returns False for non-existent document"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    exists = await TestProduct.exists("nonexistent_id")
    assert exists is False


@pytest.mark.asyncio
async def test_delete_self(redis_client):
    """Test delete_self() method"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    product = TestProduct(name="Test", price=10.0)
    await product.insert()

    # Verify it exists
    exists_before = await TestProduct.exists(product.id)
    assert exists_before is True

    # Delete
    await product.delete_self()

    # Verify it's gone
    exists_after = await TestProduct.exists(product.id)
    assert exists_after is False


@pytest.mark.asyncio
async def test_delete_all(redis_client):
    """Test delete_all() method"""
    await init_beanis(database=redis_client, document_models=[TestProduct])

    # Create multiple products
    products = [
        TestProduct(name=f"Product {i}", price=float(i))
        for i in range(5)
    ]
    await TestProduct.insert_many(products)

    # Verify count
    count_before = await TestProduct.count()
    assert count_before == 5

    # Delete all
    await TestProduct.delete_all()

    # Verify all gone
    count_after = await TestProduct.count()
    assert count_after == 0
