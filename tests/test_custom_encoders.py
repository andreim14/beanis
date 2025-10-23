"""
Test custom encoder/decoder registration system
"""
import pytest
import base64
import pickle
from typing import Any
from beanis import (
    Document,
    init_beanis,
    register_encoder,
    register_decoder,
    register_type,
    CustomEncoderRegistry,
)


# Custom type for testing
class Point3D:
    """Custom 3D point class (not a Pydantic model)"""
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return (
            isinstance(other, Point3D) and
            self.x == other.x and
            self.y == other.y and
            self.z == other.z
        )

    def __repr__(self):
        return f"Point3D({self.x}, {self.y}, {self.z})"


class CustomEncoderDoc(Document):
    """Document with custom encoded types"""
    name: str
    point: Any  # Will be Point3D
    data: Any  # Could be bytes or other custom type

    class Settings:
        name = "custom_encoder_docs"


# ============================================================================
# Tests
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clean up custom encoder registry before/after each test"""
    CustomEncoderRegistry.clear()
    yield
    CustomEncoderRegistry.clear()


@pytest.mark.asyncio
async def test_register_encoder_decoder_decorators(redis_client):
    """Test registering custom encoders/decoders using decorators"""
    await init_beanis(database=redis_client, document_models=[CustomEncoderDoc])

    # Register encoder and decoder for Point3D
    @register_encoder(Point3D)
    def encode_point(point: Point3D) -> str:
        return f"{point.x},{point.y},{point.z}"

    @register_decoder(Point3D)
    def decode_point(data: str) -> Point3D:
        x, y, z = map(float, data.split(','))
        return Point3D(x, y, z)

    # Create document with custom type
    doc = CustomEncoderDoc(
        name="Point Document",
        point=Point3D(1.5, 2.5, 3.5),
        data="test"
    )
    await doc.insert()

    # Retrieve and verify
    found = await CustomEncoderDoc.get(doc.id)
    assert found is not None
    assert found.name == "Point Document"
    assert isinstance(found.point, Point3D)
    assert found.point == Point3D(1.5, 2.5, 3.5)


@pytest.mark.asyncio
async def test_register_type_function(redis_client):
    """Test registering encoder/decoder pair with register_type()"""
    await init_beanis(database=redis_client, document_models=[CustomEncoderDoc])

    # Register both at once
    register_type(
        Point3D,
        encoder=lambda p: f"{p.x},{p.y},{p.z}",
        decoder=lambda s: Point3D(*map(float, s.split(',')))
    )

    doc = CustomEncoderDoc(
        name="Point Document",
        point=Point3D(10.0, 20.0, 30.0),
        data="test"
    )
    await doc.insert()

    found = await CustomEncoderDoc.get(doc.id)
    assert found.point == Point3D(10.0, 20.0, 30.0)


@pytest.mark.asyncio
async def test_bytes_custom_encoder(redis_client):
    """Test custom encoder for bytes (base64 encoding)"""
    await init_beanis(database=redis_client, document_models=[CustomEncoderDoc])

    # Register bytes encoder/decoder
    register_type(
        bytes,
        encoder=lambda b: base64.b64encode(b).decode('utf-8'),
        decoder=lambda s: base64.b64decode(s.encode('utf-8'))
    )

    doc = CustomEncoderDoc(
        name="Bytes Document",
        point="test",
        data=b"Hello, World! \x00\x01\x02\xff"
    )
    await doc.insert()

    found = await CustomEncoderDoc.get(doc.id)
    assert found.data == b"Hello, World! \x00\x01\x02\xff"
    assert isinstance(found.data, bytes)


@pytest.mark.asyncio
async def test_numpy_array_custom_encoder(redis_client):
    """Test custom encoder for numpy arrays"""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy not installed")

    await init_beanis(database=redis_client, document_models=[CustomEncoderDoc])

    # Register numpy encoder/decoder
    register_type(
        np.ndarray,
        encoder=lambda arr: base64.b64encode(pickle.dumps(arr)).decode('utf-8'),
        decoder=lambda s: pickle.loads(base64.b64decode(s.encode('utf-8')))
    )

    doc = CustomEncoderDoc(
        name="NumPy Document",
        point="test",
        data=np.array([[1, 2, 3], [4, 5, 6]])
    )
    await doc.insert()

    found = await CustomEncoderDoc.get(doc.id)
    assert isinstance(found.data, np.ndarray)
    np.testing.assert_array_equal(found.data, np.array([[1, 2, 3], [4, 5, 6]]))


@pytest.mark.asyncio
async def test_torch_tensor_custom_encoder(redis_client):
    """Test custom encoder for PyTorch tensors"""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")

    await init_beanis(database=redis_client, document_models=[CustomEncoderDoc])

    # Register torch encoder/decoder
    register_type(
        torch.Tensor,
        encoder=lambda t: base64.b64encode(pickle.dumps(t)).decode('utf-8'),
        decoder=lambda s: pickle.loads(base64.b64decode(s.encode('utf-8')))
    )

    doc = CustomEncoderDoc(
        name="Torch Document",
        point="test",
        data=torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )
    await doc.insert()

    found = await CustomEncoderDoc.get(doc.id)
    assert isinstance(found.data, torch.Tensor)
    assert torch.equal(found.data, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


@pytest.mark.asyncio
async def test_multiple_custom_types(redis_client):
    """Test multiple custom types in same document"""
    await init_beanis(database=redis_client, document_models=[CustomEncoderDoc])

    # Register Point3D
    register_type(
        Point3D,
        encoder=lambda p: f"{p.x},{p.y},{p.z}",
        decoder=lambda s: Point3D(*map(float, s.split(',')))
    )

    # Register bytes
    register_type(
        bytes,
        encoder=lambda b: base64.b64encode(b).decode('utf-8'),
        decoder=lambda s: base64.b64decode(s.encode('utf-8'))
    )

    doc = CustomEncoderDoc(
        name="Mixed Document",
        point=Point3D(1.0, 2.0, 3.0),
        data=b"binary data"
    )
    await doc.insert()

    found = await CustomEncoderDoc.get(doc.id)
    assert found.point == Point3D(1.0, 2.0, 3.0)
    assert found.data == b"binary data"


@pytest.mark.asyncio
async def test_get_many_with_custom_encoders(redis_client):
    """Test get_many() with custom encoded types"""
    await init_beanis(database=redis_client, document_models=[CustomEncoderDoc])

    register_type(
        Point3D,
        encoder=lambda p: f"{p.x},{p.y},{p.z}",
        decoder=lambda s: Point3D(*map(float, s.split(',')))
    )

    # Insert multiple documents
    docs = [
        CustomEncoderDoc(
            name=f"Point {i}",
            point=Point3D(float(i), float(i*2), float(i*3)),
            data=f"data{i}"
        )
        for i in range(5)
    ]

    for doc in docs:
        await doc.insert()

    # Get all at once
    ids = [doc.id for doc in docs]
    found = await CustomEncoderDoc.get_many(ids)

    # Verify all custom types preserved
    assert len(found) == 5
    for i, doc in enumerate(found):
        assert doc.name == f"Point {i}"
        assert doc.point == Point3D(float(i), float(i*2), float(i*3))


@pytest.mark.asyncio
async def test_update_with_custom_encoder(redis_client):
    """Test updating documents with custom encoded types"""
    await init_beanis(database=redis_client, document_models=[CustomEncoderDoc])

    register_type(
        Point3D,
        encoder=lambda p: f"{p.x},{p.y},{p.z}",
        decoder=lambda s: Point3D(*map(float, s.split(',')))
    )

    doc = CustomEncoderDoc(
        name="Original",
        point=Point3D(1.0, 2.0, 3.0),
        data="original"
    )
    await doc.insert()

    # Update the point
    doc.point = Point3D(10.0, 20.0, 30.0)
    await doc.save()

    # Retrieve and verify
    found = await CustomEncoderDoc.get(doc.id)
    assert found.point == Point3D(10.0, 20.0, 30.0)


@pytest.mark.asyncio
async def test_encoder_without_decoder(redis_client):
    """Test that encoder without decoder still works for writing"""
    await init_beanis(database=redis_client, document_models=[CustomEncoderDoc])

    # Only register encoder (no decoder)
    @register_encoder(Point3D)
    def encode_point(point: Point3D) -> str:
        return f"{point.x},{point.y},{point.z}"

    doc = CustomEncoderDoc(
        name="Encoder Only",
        point=Point3D(1.0, 2.0, 3.0),
        data="test"
    )
    await doc.insert()  # Should work

    # Reading will return the string (not decoded)
    found = await CustomEncoderDoc.get(doc.id)
    assert found.point == "1.0,2.0,3.0"  # String, not Point3D


@pytest.mark.asyncio
async def test_registry_isolation(redis_client):
    """Test that registry is properly isolated between tests"""
    await init_beanis(database=redis_client, document_models=[CustomEncoderDoc])

    # Should not have any encoders from previous tests
    assert CustomEncoderRegistry.get_encoder(Point3D) is None
    assert CustomEncoderRegistry.get_decoder(Point3D) is None

    # Register new encoder
    register_type(
        Point3D,
        encoder=lambda p: f"NEW:{p.x},{p.y},{p.z}",
        decoder=lambda s: Point3D(*map(float, s.replace("NEW:", "").split(',')))
    )

    doc = CustomEncoderDoc(
        name="Test",
        point=Point3D(1.0, 2.0, 3.0),
        data="test"
    )
    await doc.insert()

    found = await CustomEncoderDoc.get(doc.id)
    assert found.point == Point3D(1.0, 2.0, 3.0)


@pytest.mark.asyncio
async def test_complex_custom_type(redis_client):
    """Test custom encoder for complex nested structures"""
    await init_beanis(database=redis_client, document_models=[CustomEncoderDoc])

    class ComplexObject:
        def __init__(self, data: dict):
            self.data = data

        def __eq__(self, other):
            return isinstance(other, ComplexObject) and self.data == other.data

    register_type(
        ComplexObject,
        encoder=lambda obj: base64.b64encode(pickle.dumps(obj.data)).decode(),
        decoder=lambda s: ComplexObject(pickle.loads(base64.b64decode(s)))
    )

    doc = CustomEncoderDoc(
        name="Complex",
        point="test",
        data=ComplexObject({
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42
        })
    )
    await doc.insert()

    found = await CustomEncoderDoc.get(doc.id)
    assert isinstance(found.data, ComplexObject)
    assert found.data.data == {"nested": {"key": "value"}, "list": [1, 2, 3], "number": 42}
