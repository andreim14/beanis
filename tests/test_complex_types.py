"""
Test complex data types serialization/deserialization
Tests tuples, sets, nested structures, enums, dates, etc.
"""
import pytest
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Set, Tuple, Any
from uuid import UUID, uuid4

from beanis import Document, init_beanis
from pydantic import Field, BaseModel


# Test models with various complex types
class Color(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Coordinates(BaseModel):
    lat: float
    lon: float


class Address(BaseModel):
    street: str
    city: str
    zip_code: str
    coordinates: Optional[Coordinates] = None


class ComplexProduct(Document):
    """Document with all complex types"""
    name: str

    # Nested models
    address: Optional[Address] = None

    # Collections
    tags: List[str] = Field(default_factory=list)
    categories: Set[str] = Field(default_factory=set)
    dimensions: Tuple[int, int, int] = (0, 0, 0)

    # Dict types
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prices_by_region: Dict[str, float] = Field(default_factory=dict)

    # Special types
    color: Optional[Color] = None
    created_at: Optional[datetime] = None
    manufacture_date: Optional[date] = None
    processing_time: Optional[timedelta] = None
    product_id: Optional[UUID] = None

    # Numeric types
    price: Decimal = Decimal("0.00")
    weight: float = 0.0
    stock: int = 0

    # Nested collections
    variants: List[Dict[str, Any]] = Field(default_factory=list)
    related_products: List[List[str]] = Field(default_factory=list)

    class Settings:
        name = "complex_products"


class NestedStructure(BaseModel):
    """Deeply nested structure"""
    level: int
    data: Dict[str, Any]
    children: Optional[List['NestedStructure']] = None


class DeepNestingProduct(Document):
    """Document with deep nesting"""
    name: str
    structure: Optional[NestedStructure] = None

    class Settings:
        name = "deep_nesting_products"


# Enable forward references for recursive models
NestedStructure.model_rebuild()


# ============================================================================
# Tests
# ============================================================================

@pytest.mark.asyncio
async def test_tuple_serialization(redis_client):
    """Test tuple serialization and deserialization"""
    await init_beanis(database=redis_client, document_models=[ComplexProduct])

    product = ComplexProduct(
        name="Box",
        dimensions=(10, 20, 30)
    )
    await product.insert()

    # Retrieve and verify
    found = await ComplexProduct.get(product.id)
    assert found is not None
    assert found.dimensions == (10, 20, 30)
    assert isinstance(found.dimensions, tuple)
    assert len(found.dimensions) == 3


@pytest.mark.asyncio
async def test_set_serialization(redis_client):
    """Test set serialization and deserialization"""
    await init_beanis(database=redis_client, document_models=[ComplexProduct])

    product = ComplexProduct(
        name="Tagged Product",
        categories={"electronics", "gadgets", "new"}
    )
    await product.insert()

    # Retrieve and verify
    found = await ComplexProduct.get(product.id)
    assert found is not None
    assert found.categories == {"electronics", "gadgets", "new"}
    assert isinstance(found.categories, set)


@pytest.mark.asyncio
async def test_list_serialization(redis_client):
    """Test list serialization with various types"""
    await init_beanis(database=redis_client, document_models=[ComplexProduct])

    product = ComplexProduct(
        name="Tagged Product",
        tags=["new", "sale", "featured"]
    )
    await product.insert()

    # Retrieve and verify
    found = await ComplexProduct.get(product.id)
    assert found is not None
    assert found.tags == ["new", "sale", "featured"]
    assert isinstance(found.tags, list)


@pytest.mark.asyncio
async def test_dict_serialization(redis_client):
    """Test dict serialization with various value types"""
    await init_beanis(database=redis_client, document_models=[ComplexProduct])

    product = ComplexProduct(
        name="Regional Product",
        prices_by_region={
            "US": 99.99,
            "EU": 89.99,
            "UK": 79.99
        },
        metadata={
            "manufacturer": "ACME Corp",
            "warranty_years": 2,
            "is_refurbished": False,
            "features": ["wifi", "bluetooth"]
        }
    )
    await product.insert()

    # Retrieve and verify
    found = await ComplexProduct.get(product.id)
    assert found is not None
    assert found.prices_by_region == {"US": 99.99, "EU": 89.99, "UK": 79.99}
    assert found.metadata["manufacturer"] == "ACME Corp"
    assert found.metadata["warranty_years"] == 2
    assert found.metadata["is_refurbished"] is False
    assert found.metadata["features"] == ["wifi", "bluetooth"]


@pytest.mark.asyncio
async def test_enum_serialization(redis_client):
    """Test enum serialization"""
    await init_beanis(database=redis_client, document_models=[ComplexProduct])

    product = ComplexProduct(
        name="Colored Product",
        color=Color.RED
    )
    await product.insert()

    # Retrieve and verify
    found = await ComplexProduct.get(product.id)
    assert found is not None
    assert found.color == Color.RED
    assert isinstance(found.color, Color)


@pytest.mark.asyncio
async def test_datetime_serialization(redis_client):
    """Test datetime, date, and timedelta serialization"""
    await init_beanis(database=redis_client, document_models=[ComplexProduct])

    now = datetime(2024, 1, 15, 10, 30, 45)
    today = date(2024, 1, 15)
    processing = timedelta(hours=2, minutes=30)

    product = ComplexProduct(
        name="Time-sensitive Product",
        created_at=now,
        manufacture_date=today,
        processing_time=processing
    )
    await product.insert()

    # Retrieve and verify
    found = await ComplexProduct.get(product.id)
    assert found is not None
    assert found.created_at == now
    assert found.manufacture_date == today
    assert found.processing_time == processing


@pytest.mark.asyncio
async def test_uuid_serialization(redis_client):
    """Test UUID serialization"""
    await init_beanis(database=redis_client, document_models=[ComplexProduct])

    product_uuid = uuid4()

    product = ComplexProduct(
        name="UUID Product",
        product_id=product_uuid
    )
    await product.insert()

    # Retrieve and verify
    found = await ComplexProduct.get(product.id)
    assert found is not None
    assert found.product_id == product_uuid
    assert isinstance(found.product_id, UUID)


@pytest.mark.asyncio
async def test_decimal_serialization(redis_client):
    """Test Decimal serialization for precise money values"""
    await init_beanis(database=redis_client, document_models=[ComplexProduct])

    product = ComplexProduct(
        name="Expensive Product",
        price=Decimal("12345.67")
    )
    await product.insert()

    # Retrieve and verify
    found = await ComplexProduct.get(product.id)
    assert found is not None
    assert found.price == Decimal("12345.67")
    assert isinstance(found.price, Decimal)


@pytest.mark.asyncio
async def test_nested_model_serialization(redis_client):
    """Test nested Pydantic model serialization"""
    await init_beanis(database=redis_client, document_models=[ComplexProduct])

    product = ComplexProduct(
        name="Product with Address",
        address=Address(
            street="123 Main St",
            city="New York",
            zip_code="10001",
            coordinates=Coordinates(lat=40.7128, lon=-74.0060)
        )
    )
    await product.insert()

    # Retrieve and verify
    found = await ComplexProduct.get(product.id)
    assert found is not None
    assert found.address is not None
    assert found.address.street == "123 Main St"
    assert found.address.city == "New York"
    assert found.address.coordinates.lat == 40.7128
    assert found.address.coordinates.lon == -74.0060


@pytest.mark.asyncio
async def test_nested_collections(redis_client):
    """Test nested collections (list of dicts, list of lists)"""
    await init_beanis(database=redis_client, document_models=[ComplexProduct])

    product = ComplexProduct(
        name="Product with Variants",
        variants=[
            {"size": "S", "color": "red", "stock": 10},
            {"size": "M", "color": "blue", "stock": 5},
            {"size": "L", "color": "green", "stock": 0}
        ],
        related_products=[
            ["prod1", "prod2"],
            ["prod3", "prod4", "prod5"]
        ]
    )
    await product.insert()

    # Retrieve and verify
    found = await ComplexProduct.get(product.id)
    assert found is not None
    assert len(found.variants) == 3
    assert found.variants[0]["size"] == "S"
    assert found.variants[0]["stock"] == 10
    assert found.related_products == [["prod1", "prod2"], ["prod3", "prod4", "prod5"]]


@pytest.mark.asyncio
async def test_deep_nesting(redis_client):
    """Test deeply nested structures"""
    await init_beanis(database=redis_client, document_models=[DeepNestingProduct])

    product = DeepNestingProduct(
        name="Deeply Nested Product",
        structure=NestedStructure(
            level=1,
            data={"key1": "value1"},
            children=[
                NestedStructure(
                    level=2,
                    data={"key2": "value2"},
                    children=[
                        NestedStructure(
                            level=3,
                            data={"key3": "value3"},
                            children=None
                        )
                    ]
                )
            ]
        )
    )
    await product.insert()

    # Retrieve and verify
    found = await DeepNestingProduct.get(product.id)
    assert found is not None
    assert found.structure.level == 1
    assert found.structure.children[0].level == 2
    assert found.structure.children[0].children[0].level == 3
    assert found.structure.children[0].children[0].data["key3"] == "value3"


@pytest.mark.asyncio
async def test_empty_collections(redis_client):
    """Test empty collections serialize correctly"""
    await init_beanis(database=redis_client, document_models=[ComplexProduct])

    product = ComplexProduct(
        name="Empty Collections Product",
        tags=[],
        categories=set(),
        metadata={},
        variants=[]
    )
    await product.insert()

    # Retrieve and verify
    found = await ComplexProduct.get(product.id)
    assert found is not None
    assert found.tags == []
    assert found.categories == set()
    assert found.metadata == {}
    assert found.variants == []


@pytest.mark.asyncio
async def test_none_values(redis_client):
    """Test None values for optional fields"""
    await init_beanis(database=redis_client, document_models=[ComplexProduct])

    product = ComplexProduct(
        name="Product with Nones",
        address=None,
        color=None,
        created_at=None,
        product_id=None
    )
    await product.insert()

    # Retrieve and verify
    found = await ComplexProduct.get(product.id)
    assert found is not None
    assert found.address is None
    assert found.color is None
    assert found.created_at is None
    assert found.product_id is None


@pytest.mark.asyncio
async def test_mixed_complex_types(redis_client):
    """Test document with all complex types at once"""
    await init_beanis(database=redis_client, document_models=[ComplexProduct])

    product = ComplexProduct(
        name="Kitchen Blender",
        address=Address(
            street="456 Oak Ave",
            city="San Francisco",
            zip_code="94102",
            coordinates=Coordinates(lat=37.7749, lon=-122.4194)
        ),
        tags=["kitchen", "appliance", "sale"],
        categories={"electronics", "home", "kitchen"},
        dimensions=(15, 20, 25),
        metadata={
            "brand": "BlendTech",
            "model": "BT-3000",
            "wattage": 1200,
            "has_warranty": True
        },
        prices_by_region={
            "US": 149.99,
            "EU": 139.99,
            "UK": 129.99
        },
        color=Color.BLUE,
        created_at=datetime(2024, 1, 15, 10, 0, 0),
        manufacture_date=date(2024, 1, 1),
        processing_time=timedelta(hours=1, minutes=30),
        product_id=uuid4(),
        price=Decimal("149.99"),
        weight=2.5,
        stock=50,
        variants=[
            {"color": "blue", "stock": 30},
            {"color": "red", "stock": 20}
        ],
        related_products=[["blender-1", "blender-2"], ["mixer-1"]]
    )
    await product.insert()

    # Retrieve and verify everything
    found = await ComplexProduct.get(product.id)
    assert found is not None
    assert found.name == "Kitchen Blender"
    assert found.address.city == "San Francisco"
    assert found.tags == ["kitchen", "appliance", "sale"]
    assert found.categories == {"electronics", "home", "kitchen"}
    assert found.dimensions == (15, 20, 25)
    assert found.metadata["brand"] == "BlendTech"
    assert found.prices_by_region["US"] == 149.99
    assert found.color == Color.BLUE
    assert found.created_at.year == 2024
    assert found.price == Decimal("149.99")
    assert len(found.variants) == 2
    assert len(found.related_products) == 2


@pytest.mark.asyncio
async def test_update_complex_types(redis_client):
    """Test updating documents with complex types"""
    await init_beanis(database=redis_client, document_models=[ComplexProduct])

    product = ComplexProduct(
        name="Updateable Product",
        tags=["old"],
        categories={"outdated"}
    )
    await product.insert()

    # Update with new complex values
    product.tags = ["new", "updated"]
    product.categories = {"fresh", "modern"}
    product.dimensions = (1, 2, 3)
    await product.save()

    # Retrieve and verify
    found = await ComplexProduct.get(product.id)
    assert found is not None
    assert found.tags == ["new", "updated"]
    assert found.categories == {"fresh", "modern"}
    assert found.dimensions == (1, 2, 3)


@pytest.mark.asyncio
async def test_get_many_complex_types(redis_client):
    """Test get_many with documents containing complex types"""
    await init_beanis(database=redis_client, document_models=[ComplexProduct])

    products = [
        ComplexProduct(
            name=f"Product {i}",
            tags=[f"tag{i}"],
            categories={f"cat{i}"},
            dimensions=(i, i*2, i*3),
            metadata={"index": i}
        )
        for i in range(5)
    ]

    for p in products:
        await p.insert()

    # Get all at once
    ids = [p.id for p in products]
    found = await ComplexProduct.get_many(ids)

    # Verify all complex types preserved
    assert len(found) == 5
    for i, p in enumerate(found):
        assert p.name == f"Product {i}"
        assert p.tags == [f"tag{i}"]
        assert p.categories == {f"cat{i}"}
        assert p.dimensions == (i, i*2, i*3)
        assert p.metadata["index"] == i
