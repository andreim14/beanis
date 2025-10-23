"""
Comprehensive tests for batch operations to improve coverage
Tests insert_many, get_many, delete_many with various scenarios
"""
import pytest
from typing import Optional
from pydantic import BaseModel

from beanis import Document, init_beanis, Indexed


class Category(BaseModel):
    name: str
    code: str


class Product(Document):
    name: str
    price: Indexed(float)
    category: Optional[Category] = None
    tags: Optional[list[str]] = None
    stock: int = 0

    class Settings:
        key_prefix = "BatchProduct"


@pytest.mark.asyncio
async def test_insert_many_empty_list(redis_client):
    """Test insert_many with empty list"""
    await init_beanis(database=redis_client, document_models=[Product])

    result = await Product.insert_many([])
    assert result == []


@pytest.mark.asyncio
async def test_insert_many_single_item(redis_client):
    """Test insert_many with single item"""
    await init_beanis(database=redis_client, document_models=[Product])

    products = [Product(name="Single", price=10.0)]
    result = await Product.insert_many(products)

    assert len(result) == 1
    assert result[0].id is not None


@pytest.mark.asyncio
async def test_insert_many_with_nested_models(redis_client):
    """Test insert_many with complex nested Pydantic models"""
    await init_beanis(database=redis_client, document_models=[Product])

    products = [
        Product(
            name=f"Product {i}",
            price=float(i * 10),
            category=Category(name=f"Cat {i}", code=f"C{i}")
        )
        for i in range(1, 6)
    ]

    await Product.insert_many(products)

    # Verify they were inserted with categories intact
    found = await Product.get(products[0].id)
    assert found.category is not None
    assert found.category.name == "Cat 1"
    assert found.category.code == "C1"


@pytest.mark.asyncio
async def test_insert_many_with_lists(redis_client):
    """Test insert_many with list fields"""
    await init_beanis(database=redis_client, document_models=[Product])

    products = [
        Product(name=f"Prod {i}", price=10.0, tags=[f"tag{j}" for j in range(1, 4)])
        for i in range(1, 6)
    ]

    await Product.insert_many(products)

    found = await Product.get(products[2].id)
    assert found.tags == ["tag1", "tag2", "tag3"]


@pytest.mark.asyncio
async def test_insert_many_preserves_ids(redis_client):
    """Test insert_many preserves explicit IDs"""
    await init_beanis(database=redis_client, document_models=[Product])

    products = [
        Product(id=f"custom-{i}", name=f"Product {i}", price=10.0)
        for i in range(1, 4)
    ]

    await Product.insert_many(products)

    # Verify explicit IDs were used
    found1 = await Product.get("custom-1")
    found2 = await Product.get("custom-2")
    found3 = await Product.get("custom-3")

    assert found1 is not None
    assert found2 is not None
    assert found3 is not None


@pytest.mark.asyncio
async def test_insert_many_autogenerate_ids(redis_client):
    """Test insert_many auto-generates IDs when not provided"""
    await init_beanis(database=redis_client, document_models=[Product])

    products = [
        Product(name=f"Product {i}", price=10.0)
        for i in range(1, 6)
    ]

    # IDs should be None initially
    assert all(p.id is None or p.id == "" for p in products)

    result = await Product.insert_many(products)

    # IDs should be auto-generated
    assert all(p.id is not None and len(p.id) > 0 for p in result)


@pytest.mark.asyncio
async def test_get_many_all_exist(redis_client):
    """Test get_many when all documents exist"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert products
    ids = [f"getmany-{i}" for i in range(1, 6)]
    products = [
        Product(id=id, name=f"Product {id}", price=float(i * 10))
        for i, id in enumerate(ids, 1)
    ]
    await Product.insert_many(products)

    # Get them all
    found = await Product.get_many(ids)

    assert len(found) == 5
    assert all(p is not None for p in found)
    assert [p.id for p in found] == ids


@pytest.mark.asyncio
async def test_get_many_some_missing(redis_client):
    """Test get_many with mix of existing and non-existing documents"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert only some
    product1 = Product(id="exists-1", name="Product 1", price=10.0)
    product3 = Product(id="exists-3", name="Product 3", price=30.0)
    await product1.insert()
    await product3.insert()

    # Try to get mix of existing and non-existing
    ids = ["exists-1", "missing-2", "exists-3", "missing-4"]
    found = await Product.get_many(ids)

    assert len(found) == 4
    assert found[0] is not None
    assert found[0].id == "exists-1"
    assert found[1] is None
    assert found[2] is not None
    assert found[2].id == "exists-3"
    assert found[3] is None


@pytest.mark.asyncio
async def test_get_many_empty_list(redis_client):
    """Test get_many with empty list"""
    await init_beanis(database=redis_client, document_models=[Product])

    found = await Product.get_many([])
    assert found == []


@pytest.mark.asyncio
async def test_get_many_single_id(redis_client):
    """Test get_many with single ID"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="single", name="Single Product", price=25.0)
    await product.insert()

    found = await Product.get_many(["single"])

    assert len(found) == 1
    assert found[0].name == "Single Product"


@pytest.mark.asyncio
async def test_get_many_preserves_order(redis_client):
    """Test get_many preserves order of requested IDs"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert in one order
    products = [
        Product(id=f"p{i}", name=f"Product {i}", price=10.0)
        for i in range(1, 6)
    ]
    await Product.insert_many(products)

    # Request in different order
    requested_ids = ["p3", "p1", "p5", "p2", "p4"]
    found = await Product.get_many(requested_ids)

    # Should match requested order
    assert [p.id for p in found] == requested_ids


@pytest.mark.asyncio
async def test_delete_many_all_exist(redis_client):
    """Test delete_many when all documents exist"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert products
    ids = [f"del-{i}" for i in range(1, 11)]
    products = [
        Product(id=id, name=f"Product {id}", price=10.0)
        for id in ids
    ]
    await Product.insert_many(products)

    # Delete half of them
    to_delete = ids[:5]
    deleted_count = await Product.delete_many(to_delete)

    assert deleted_count == 5

    # Verify deleted
    for id in to_delete:
        assert await Product.exists(id) is False

    # Verify others still exist
    for id in ids[5:]:
        assert await Product.exists(id) is True


@pytest.mark.asyncio
async def test_delete_many_some_missing(redis_client):
    """Test delete_many with mix of existing and non-existing IDs"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert only 3 products
    products = [
        Product(id=f"exists-{i}", name=f"Product {i}", price=10.0)
        for i in range(1, 4)
    ]
    await Product.insert_many(products)

    # Try to delete both existing and non-existing
    to_delete = ["exists-1", "missing-1", "exists-2", "missing-2", "exists-3"]
    deleted_count = await Product.delete_many(to_delete)

    # Should delete only the 3 that existed
    assert deleted_count == 3

    # Verify all are gone
    assert await Product.count() == 0


@pytest.mark.asyncio
async def test_delete_many_empty_list(redis_client):
    """Test delete_many with empty list"""
    await init_beanis(database=redis_client, document_models=[Product])

    deleted_count = await Product.delete_many([])
    assert deleted_count == 0


@pytest.mark.asyncio
async def test_delete_many_none_exist(redis_client):
    """Test delete_many when none of the IDs exist"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Don't insert anything
    deleted_count = await Product.delete_many(["fake-1", "fake-2", "fake-3"])

    assert deleted_count == 0


@pytest.mark.asyncio
async def test_insert_many_large_batch(redis_client):
    """Test insert_many with large batch"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert 1000 documents
    products = [
        Product(name=f"Product {i}", price=float(i % 100), stock=i)
        for i in range(1000)
    ]

    result = await Product.insert_many(products)

    assert len(result) == 1000

    # Verify count
    count = await Product.count()
    assert count == 1000


@pytest.mark.asyncio
async def test_get_many_large_batch(redis_client):
    """Test get_many with large batch"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert 500 documents
    ids = [f"large-{i}" for i in range(500)]
    products = [
        Product(id=id, name=f"Product {id}", price=10.0)
        for id in ids
    ]
    await Product.insert_many(products)

    # Get all 500
    found = await Product.get_many(ids)

    assert len(found) == 500
    assert all(p is not None for p in found)


@pytest.mark.asyncio
async def test_delete_many_large_batch(redis_client):
    """Test delete_many with large batch"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert 500 documents
    ids = [f"del-large-{i}" for i in range(500)]
    products = [
        Product(id=id, name=f"Product {id}", price=10.0)
        for id in ids
    ]
    await Product.insert_many(products)

    # Delete all 500
    deleted_count = await Product.delete_many(ids)

    assert deleted_count == 500
    assert await Product.count() == 0


@pytest.mark.asyncio
async def test_batch_operations_with_indexed_fields(redis_client):
    """Test batch operations properly handle indexed fields"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert products with indexed price field
    products = [
        Product(name=f"Product {i}", price=float(i * 10), stock=i)
        for i in range(1, 11)
    ]
    result = await Product.insert_many(products)

    # Verify all were inserted
    assert len(result) == 10

    # Count all products
    count_before = await Product.count()
    assert count_before == 10

    # Delete half
    ids_to_delete = [p.id for p in result[:5]]
    deleted = await Product.delete_many(ids_to_delete)
    assert deleted == 5

    # Verify count updated
    count_after = await Product.count()
    assert count_after == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
