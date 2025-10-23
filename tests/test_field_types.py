"""
Tests for various field types and edge cases to improve coverage
"""
import pytest
from typing import Optional, List, Dict, Set, Tuple
from decimal import Decimal
from datetime import datetime, date, time, timedelta
from uuid import UUID, uuid4
from enum import Enum

from beanis import Document, init_beanis


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class ProductWithAllTypes(Document):
    # Basic types
    name: str
    price: float
    stock: int
    is_active: bool

    # Optional types
    description: Optional[str] = None
    rating: Optional[float] = None

    # Special types
    decimal_price: Optional[Decimal] = None
    uuid_id: Optional[UUID] = None
    color: Optional[Color] = None

    # Date/time types
    created_at: Optional[datetime] = None
    launch_date: Optional[date] = None
    launch_time: Optional[time] = None
    duration: Optional[timedelta] = None

    # Collections
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, str]] = None
    unique_tags: Optional[Set[str]] = None
    coordinates: Optional[Tuple[float, float]] = None

    class Settings:
        key_prefix = "AllTypes"


@pytest.mark.asyncio
async def test_string_field(redis_client):
    """Test string field"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    product = ProductWithAllTypes(
        name="Test Product",
        price=10.0,
        stock=5,
        is_active=True
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.name == "Test Product"
    assert isinstance(found.name, str)


@pytest.mark.asyncio
async def test_float_field(redis_client):
    """Test float field"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    product = ProductWithAllTypes(
        name="Test",
        price=19.99,
        stock=10,
        is_active=True
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.price == 19.99
    assert isinstance(found.price, float)


@pytest.mark.asyncio
async def test_int_field(redis_client):
    """Test integer field"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    product = ProductWithAllTypes(
        name="Test",
        price=10.0,
        stock=42,
        is_active=True
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.stock == 42
    assert isinstance(found.stock, int)


@pytest.mark.asyncio
async def test_bool_field(redis_client):
    """Test boolean field"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    product1 = ProductWithAllTypes(
        id="bool-true",
        name="Active",
        price=10.0,
        stock=5,
        is_active=True
    )
    product2 = ProductWithAllTypes(
        id="bool-false",
        name="Inactive",
        price=10.0,
        stock=0,
        is_active=False
    )

    await product1.insert()
    await product2.insert()

    found1 = await ProductWithAllTypes.get("bool-true")
    found2 = await ProductWithAllTypes.get("bool-false")

    assert found1.is_active is True
    assert found2.is_active is False


@pytest.mark.asyncio
async def test_optional_field_none(redis_client):
    """Test optional field with None value"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    product = ProductWithAllTypes(
        name="Test",
        price=10.0,
        stock=5,
        is_active=True,
        description=None  # Explicitly None
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.description is None


@pytest.mark.asyncio
async def test_optional_field_with_value(redis_client):
    """Test optional field with actual value"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    product = ProductWithAllTypes(
        name="Test",
        price=10.0,
        stock=5,
        is_active=True,
        description="A great product"
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.description == "A great product"


@pytest.mark.asyncio
async def test_decimal_field(redis_client):
    """Test Decimal field"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    product = ProductWithAllTypes(
        name="Test",
        price=10.0,
        stock=5,
        is_active=True,
        decimal_price=Decimal("19.99")
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.decimal_price == Decimal("19.99")


@pytest.mark.asyncio
async def test_uuid_field(redis_client):
    """Test UUID field"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    test_uuid = uuid4()
    product = ProductWithAllTypes(
        name="Test",
        price=10.0,
        stock=5,
        is_active=True,
        uuid_id=test_uuid
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.uuid_id == test_uuid
    assert isinstance(found.uuid_id, UUID)


@pytest.mark.asyncio
async def test_enum_field(redis_client):
    """Test Enum field"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    product = ProductWithAllTypes(
        name="Test",
        price=10.0,
        stock=5,
        is_active=True,
        color=Color.BLUE
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.color == Color.BLUE
    assert isinstance(found.color, Color)


@pytest.mark.asyncio
async def test_datetime_field(redis_client):
    """Test datetime field"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    now = datetime(2024, 1, 15, 10, 30, 45)
    product = ProductWithAllTypes(
        name="Test",
        price=10.0,
        stock=5,
        is_active=True,
        created_at=now
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.created_at == now


@pytest.mark.asyncio
async def test_date_field(redis_client):
    """Test date field"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    test_date = date(2024, 6, 15)
    product = ProductWithAllTypes(
        name="Test",
        price=10.0,
        stock=5,
        is_active=True,
        launch_date=test_date
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.launch_date == test_date


@pytest.mark.asyncio
async def test_time_field(redis_client):
    """Test time field"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    test_time = time(14, 30, 0)
    product = ProductWithAllTypes(
        name="Test",
        price=10.0,
        stock=5,
        is_active=True,
        launch_time=test_time
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.launch_time == test_time


@pytest.mark.asyncio
async def test_timedelta_field(redis_client):
    """Test timedelta field"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    test_duration = timedelta(days=7, hours=3, minutes=30)
    product = ProductWithAllTypes(
        name="Test",
        price=10.0,
        stock=5,
        is_active=True,
        duration=test_duration
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.duration == test_duration


@pytest.mark.asyncio
async def test_list_field(redis_client):
    """Test list field"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    product = ProductWithAllTypes(
        name="Test",
        price=10.0,
        stock=5,
        is_active=True,
        tags=["tag1", "tag2", "tag3"]
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.tags == ["tag1", "tag2", "tag3"]
    assert isinstance(found.tags, list)


@pytest.mark.asyncio
async def test_dict_field(redis_client):
    """Test dict field"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    product = ProductWithAllTypes(
        name="Test",
        price=10.0,
        stock=5,
        is_active=True,
        metadata={"key1": "value1", "key2": "value2"}
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.metadata == {"key1": "value1", "key2": "value2"}
    assert isinstance(found.metadata, dict)


@pytest.mark.asyncio
async def test_set_field(redis_client):
    """Test set field"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    product = ProductWithAllTypes(
        name="Test",
        price=10.0,
        stock=5,
        is_active=True,
        unique_tags={"unique1", "unique2", "unique3"}
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.unique_tags == {"unique1", "unique2", "unique3"}
    assert isinstance(found.unique_tags, set)


@pytest.mark.asyncio
async def test_tuple_field(redis_client):
    """Test tuple field"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    product = ProductWithAllTypes(
        name="Test",
        price=10.0,
        stock=5,
        is_active=True,
        coordinates=(37.7749, -122.4194)
    )
    await product.insert()

    found = await ProductWithAllTypes.get(product.id)
    assert found.coordinates == (37.7749, -122.4194)
    assert isinstance(found.coordinates, tuple)


@pytest.mark.asyncio
async def test_all_fields_together(redis_client):
    """Test document with all field types populated"""
    await init_beanis(database=redis_client, document_models=[ProductWithAllTypes])

    product = ProductWithAllTypes(
        name="Complete Product",
        price=29.99,
        stock=100,
        is_active=True,
        description="A complete product",
        rating=4.5,
        decimal_price=Decimal("29.99"),
        uuid_id=uuid4(),
        color=Color.GREEN,
        created_at=datetime.now(),
        launch_date=date.today(),
        launch_time=time(12, 0, 0),
        duration=timedelta(hours=24),
        tags=["new", "featured", "sale"],
        metadata={"brand": "TestBrand", "category": "TestCat"},
        unique_tags={"unique1", "unique2"},
        coordinates=(40.7128, -74.0060)
    )

    await product.insert()

    found = await ProductWithAllTypes.get(product.id)

    # Verify all fields
    assert found.name == "Complete Product"
    assert found.price == 29.99
    assert found.stock == 100
    assert found.is_active is True
    assert found.description == "A complete product"
    assert found.rating == 4.5
    assert found.decimal_price == Decimal("29.99")
    assert isinstance(found.uuid_id, UUID)
    assert found.color == Color.GREEN
    assert isinstance(found.created_at, datetime)
    assert isinstance(found.launch_date, date)
    assert isinstance(found.launch_time, time)
    assert isinstance(found.duration, timedelta)
    assert found.tags == ["new", "featured", "sale"]
    assert found.metadata == {"brand": "TestBrand", "category": "TestCat"}
    assert found.unique_tags == {"unique1", "unique2"}
    assert found.coordinates == (40.7128, -74.0060)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
