"""
Test what happens with unsupported types like numpy arrays, torch tensors
"""
import pytest
from typing import Any
from beanis import Document, init_beanis


class UnsupportedTypeDoc(Document):
    """Document with unsupported types"""
    name: str
    data: Any  # This could be anything!

    class Settings:
        name = "unsupported_docs"


@pytest.mark.asyncio
async def test_numpy_array_fails(redis_client):
    """Test that numpy arrays fail with clear error"""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy not installed")

    await init_beanis(database=redis_client, document_models=[UnsupportedTypeDoc])

    doc = UnsupportedTypeDoc(
        name="Numpy Doc",
        data=np.array([1, 2, 3, 4, 5])
    )

    # Should raise an error
    with pytest.raises(Exception) as exc_info:
        await doc.insert()

    # Check error message is helpful
    assert "Cannot encode" in str(exc_info.value) or "not JSON serializable" in str(exc_info.value)


@pytest.mark.asyncio
async def test_bytes_converts_to_list(redis_client):
    """Test that bytes get converted to list (not ideal, but doesn't error)"""
    await init_beanis(database=redis_client, document_models=[UnsupportedTypeDoc])

    doc = UnsupportedTypeDoc(
        name="Bytes Doc",
        data=b"raw binary data"
    )

    # Bytes don't error, but get converted to list by JSON serialization
    await doc.insert()

    # Retrieve - will be a list of integers, not bytes
    found = await UnsupportedTypeDoc.get(doc.id)
    expected_list = list(b"raw binary data")
    assert found.data == expected_list
    assert isinstance(found.data, list)

    # To properly store bytes, use custom encoder (see test_custom_encoders.py)


@pytest.mark.asyncio
async def test_custom_object_fails(redis_client):
    """Test that custom objects fail with clear error"""
    await init_beanis(database=redis_client, document_models=[UnsupportedTypeDoc])

    class CustomObject:
        def __init__(self, value):
            self.value = value

    doc = UnsupportedTypeDoc(
        name="Custom Object Doc",
        data=CustomObject(42)
    )

    # Should raise an error
    with pytest.raises(Exception) as exc_info:
        await doc.insert()

    assert "Cannot encode" in str(exc_info.value)
