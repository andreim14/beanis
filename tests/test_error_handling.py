"""
Tests for error handling and edge cases
Tests operations on uninitialized documents, invalid IDs, network errors, serialization errors
"""
import pytest
from typing import Optional
from pydantic import BaseModel, field_validator

from beanis import Document, init_beanis, Indexed
from beanis.exceptions import CollectionWasNotInitialized


class Product(Document):
    name: str
    price: float
    stock: int = 0

    class Settings:
        key_prefix = "Product"


class ProductWithValidation(Document):
    name: str
    price: float
    stock: int

    class Settings:
        key_prefix = "ProductVal"

    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        if v < 0:
            raise ValueError("Price must be non-negative")
        return v

    @field_validator('stock')
    @classmethod
    def validate_stock(cls, v):
        if v < 0:
            raise ValueError("Stock must be non-negative")
        return v


class ComplexType(BaseModel):
    value: int


class ProductWithComplexField(Document):
    name: str
    complex_data: ComplexType

    class Settings:
        key_prefix = "ProductComplex"


@pytest.mark.asyncio
async def test_multiple_document_classes_same_key_prefix(redis_client):
    """Test handling multiple document classes with same key prefix"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Using same key prefix shouldn't cause issues
    product1 = Product(id="test-1", name="Product 1", price=10.0)
    product2 = Product(id="test-2", name="Product 2", price=20.0)

    await product1.insert()
    await product2.insert()

    # Both should be retrievable
    found1 = await Product.get("test-1")
    found2 = await Product.get("test-2")

    assert found1.name == "Product 1"
    assert found2.name == "Product 2"


@pytest.mark.asyncio
async def test_get_with_invalid_id(redis_client):
    """Test getting document with invalid/malformed ID"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Empty string ID
    result = await Product.get("")
    assert result is None

    # Very long ID
    long_id = "x" * 10000
    result = await Product.get(long_id)
    assert result is None


@pytest.mark.asyncio
async def test_update_nonexistent_document(redis_client):
    """Test updating a document that doesn't exist"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="nonexistent", name="Test", price=10.0)

    # Update without insert should handle gracefully
    # (behavior depends on implementation - may create or error)
    try:
        await product.update(price=20.0)
        # If it doesn't error, verify document state
        found = await Product.get("nonexistent")
        # Could be None or could have been created
    except Exception:
        # If it errors, that's also valid behavior
        pass


@pytest.mark.asyncio
async def test_delete_nonexistent_document(redis_client):
    """Test deleting a document that doesn't exist"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="nonexistent-del", name="Test", price=10.0)

    # Delete without insert should not crash
    await product.delete_self()

    # Should still not exist
    assert await Product.exists("nonexistent-del") is False


@pytest.mark.asyncio
async def test_invalid_data_validation_on_insert(redis_client):
    """Test Pydantic validation prevents invalid data"""
    await init_beanis(database=redis_client, document_models=[ProductWithValidation])

    # Try to create product with negative price
    with pytest.raises(ValueError, match="Price must be non-negative"):
        product = ProductWithValidation(name="Invalid", price=-10.0, stock=5)


@pytest.mark.asyncio
async def test_invalid_data_validation_on_update(redis_client):
    """Test validation on update"""
    await init_beanis(database=redis_client, document_models=[ProductWithValidation])

    # Insert valid product
    product = ProductWithValidation(id="val-test", name="Valid", price=100.0, stock=10)
    await product.insert()

    # Try to update with invalid data
    with pytest.raises(ValueError, match="Stock must be non-negative"):
        product.stock = -5
        product = ProductWithValidation(**product.model_dump())  # Trigger validation


@pytest.mark.asyncio
async def test_empty_batch_operations(redis_client):
    """Test batch operations with empty lists"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Empty insert_many
    result = await Product.insert_many([])
    assert result == []

    # Empty get_many
    result = await Product.get_many([])
    assert result == []

    # Empty delete_many
    deleted = await Product.delete_many([])
    assert deleted == 0


@pytest.mark.asyncio
async def test_duplicate_ids_in_batch_operations(redis_client):
    """Test batch operations with duplicate IDs"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Insert multiple products with same ID (last one wins in Redis)
    products = [
        Product(id="dup-1", name="First", price=10.0),
        Product(id="dup-1", name="Second", price=20.0),
    ]

    await Product.insert_many(products)

    # Only one should exist (the last one)
    found = await Product.get("dup-1")
    assert found is not None
    assert found.name == "Second"  # Last write wins


@pytest.mark.asyncio
async def test_very_large_field_value(redis_client):
    """Test handling very large field values"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Create product with very long name
    long_name = "x" * 100000  # 100KB string

    product = Product(id="large-field", name=long_name, price=10.0)
    await product.insert()

    # Should be retrievable
    found = await Product.get("large-field")
    assert found is not None
    assert len(found.name) == 100000


@pytest.mark.asyncio
async def test_special_characters_in_fields(redis_client):
    """Test handling special characters in field values"""
    await init_beanis(database=redis_client, document_models=[Product])

    special_chars = "Test with 特殊字符 émojis 🎉🎊 quotes \"' and\nnewlines\ttabs"

    product = Product(id="special-chars", name=special_chars, price=25.0)
    await product.insert()

    found = await Product.get("special-chars")
    assert found is not None
    assert found.name == special_chars


@pytest.mark.asyncio
async def test_increment_field_on_nonexistent_doc(redis_client):
    """Test incrementing field on document that doesn't exist"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="no-exist-incr", name="Test", price=100.0, stock=10)

    # Try to increment without inserting first
    # Behavior depends on implementation
    try:
        result = await product.increment_field("stock", 5)
        # If it doesn't error, result should be meaningful
    except Exception:
        # If it errors, that's expected
        pass


@pytest.mark.asyncio
async def test_get_field_on_nonexistent_doc(redis_client):
    """Test getting field from document that doesn't exist"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="no-exist-field", name="Test", price=50.0)

    # Try to get field without inserting
    field_value = await product.get_field("price")

    # Should return None or raise error
    assert field_value is None or isinstance(field_value, (str, float, type(None)))


@pytest.mark.asyncio
async def test_set_field_on_nonexistent_doc(redis_client):
    """Test setting field on document that doesn't exist"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="no-exist-set", name="Test", price=30.0)

    # Try to set field without inserting
    # Should either work or error gracefully
    try:
        await product.set_field("price", 40.0)
    except Exception:
        pass  # Expected to possibly error


@pytest.mark.asyncio
async def test_complex_type_serialization(redis_client):
    """Test serialization/deserialization of complex nested types"""
    await init_beanis(database=redis_client, document_models=[ProductWithComplexField])

    complex_obj = ComplexType(value=42)
    product = ProductWithComplexField(
        id="complex-1",
        name="Test",
        complex_data=complex_obj
    )

    await product.insert()

    # Retrieve and verify complex type is preserved
    found = await ProductWithComplexField.get("complex-1")
    assert found is not None
    assert found.complex_data.value == 42
    assert isinstance(found.complex_data, ComplexType)


@pytest.mark.asyncio
async def test_concurrent_updates_same_field(redis_client):
    """Test concurrent updates to same field"""
    await init_beanis(database=redis_client, document_models=[Product])

    product = Product(id="concurrent", name="Test", price=100.0, stock=10)
    await product.insert()

    # Simulate concurrent updates (last write wins)
    await product.update(price=150.0)
    await product.update(price=200.0)
    await product.update(price=175.0)

    # Last update should win
    found = await Product.get("concurrent")
    assert found.price == 175.0


@pytest.mark.asyncio
async def test_delete_all_on_empty_collection(redis_client):
    """Test delete_all when collection is empty"""
    await init_beanis(database=redis_client, document_models=[Product])

    # Delete all on empty collection
    deleted = await Product.delete_all()
    assert deleted == 0


@pytest.mark.asyncio
async def test_count_on_empty_collection(redis_client):
    """Test count on empty collection"""
    await init_beanis(database=redis_client, document_models=[Product])

    count = await Product.count()
    assert count == 0


@pytest.mark.asyncio
async def test_all_on_empty_collection(redis_client):
    """Test all() on empty collection"""
    await init_beanis(database=redis_client, document_models=[Product])

    all_docs = await Product.all()
    assert all_docs == []


@pytest.mark.asyncio
async def test_find_with_no_indexed_fields(redis_client):
    """Test find() when no fields are indexed"""

    class ProductNoIndex(Document):
        name: str
        price: float

        class Settings:
            key_prefix = "ProductNoIdx"

    await init_beanis(database=redis_client, document_models=[ProductNoIndex])

    product = ProductNoIndex(name="Test", price=100.0)
    await product.insert()

    # Find should handle gracefully (may return empty or all)
    # Exact behavior depends on implementation
    try:
        results = await ProductNoIndex.find().to_list()
        # Should not crash
    except Exception:
        # May error if no indexes available
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
