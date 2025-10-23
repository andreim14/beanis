"""
Tests for save() method
Tests insert vs update behavior, error handling
"""
import pytest
from typing import Optional

from beanis import Document, init_beanis


class Product(Document):
    name: str
    price: float
    stock: int = 0

    class Settings:
        key_prefix = "Product"


@pytest.mark.asyncio
async def test_save_new_document(redis_client):
    """Test save() inserts a new document"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="save-1", name="New Product", price=50.0, stock=10)

    # Should not exist yet
    assert await Product.exists("save-1") is False

    # Save it
    await product.save()

    # Should exist now
    assert await Product.exists("save-1") is True

    # Verify data
    found = await Product.get("save-1")
    assert found.name == "New Product"
    assert found.price == 50.0


@pytest.mark.asyncio
async def test_save_existing_document(redis_client):
    """Test save() updates an existing document"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert a document
    product = Product(id="save-2", name="Original", price=100.0, stock=5)
    await product.insert()

    # Modify and save
    product.name = "Updated"
    product.price = 150.0
    await product.save()

    # Verify updates
    found = await Product.get("save-2")
    assert found.name == "Updated"
    assert found.price == 150.0
    assert found.stock == 5


@pytest.mark.asyncio
async def test_save_auto_generates_id(redis_client):
    """Test save() auto-generates ID if not provided"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(name="No ID", price=25.0)

    # ID should be None initially
    assert product.id is None or product.id == ""

    # Save should auto-generate ID
    await product.save()

    # ID should be set
    assert product.id is not None
    assert len(product.id) > 0

    # Should be retrievable
    found = await Product.get(product.id)
    assert found is not None
    assert found.name == "No ID"


@pytest.mark.asyncio
async def test_save_preserves_other_fields(redis_client):
    """Test save() doesn't lose fields when updating"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert with all fields
    product = Product(id="save-3", name="Full Product", price=75.0, stock=20)
    await product.insert()

    # Update only name via save
    product.name = "Modified Name"
    await product.save()

    # All fields should still exist
    found = await Product.get("save-3")
    assert found.name == "Modified Name"
    assert found.price == 75.0
    assert found.stock == 20


@pytest.mark.asyncio
async def test_save_multiple_times(redis_client):
    """Test calling save() multiple times on same document"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="save-4", name="Test", price=10.0, stock=5)

    # First save (insert)
    await product.save()
    assert await Product.exists("save-4") is True

    # Second save (update)
    product.price = 20.0
    await product.save()

    # Third save (update again)
    product.stock = 15
    await product.save()

    # Verify final state
    found = await Product.get("save-4")
    assert found.price == 20.0
    assert found.stock == 15


@pytest.mark.asyncio
async def test_save_with_ttl(redis_client):
    """Test inserting with TTL and then saving"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="save-ttl", name="Temporary", price=30.0)

    # Insert with TTL first
    await product.insert(ttl=3600)

    # Check TTL was set
    ttl = await product.get_ttl()
    assert ttl is not None
    assert ttl > 0
    assert ttl <= 3600

    # Save should work after insert
    product.price = 35.0
    await product.save()


@pytest.mark.asyncio
async def test_save_preserves_ttl(redis_client):
    """Test save() preserves TTL on existing document"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="save-ttl-2", name="Test", price=40.0)

    # Initial insert with TTL
    await product.insert(ttl=7200)

    # Save again with update
    product.price = 45.0
    await product.save()

    # TTL should still exist
    ttl = await product.get_ttl()
    assert ttl is not None
    assert ttl > 0


@pytest.mark.asyncio
async def test_save_after_get(redis_client):
    """Test modifying and saving a document retrieved with get()"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert a product
    original = Product(id="save-5", name="Original", price=60.0, stock=8)
    await original.insert()

    # Get the product
    product = await Product.get("save-5")
    assert product is not None

    # Modify and save
    product.name = "Modified After Get"
    product.price = 65.0
    await product.save()

    # Verify changes
    found = await Product.get("save-5")
    assert found.name == "Modified After Get"
    assert found.price == 65.0


@pytest.mark.asyncio
async def test_save_vs_insert_difference(redis_client):
    """Test that save() is idempotent but insert() is not"""
    await init_beanis(database=redis_client, document_models=[Product])

    product1 = Product(id="save-insert-1", name="Test", price=100.0)

    # save() twice should work fine
    await product1.save()
    await product1.save()  # Should not error

    # insert() twice should update (Redis behavior)
    product2 = Product(id="save-insert-2", name="Test", price=200.0)
    await product2.insert()

    # Modify and insert again
    product2.price = 250.0
    await product2.insert()  # This updates in Redis

    found = await Product.get("save-insert-2")
    assert found.price == 250.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
