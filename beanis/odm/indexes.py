"""
Redis-based indexing system using Sets and Sorted Sets

This module provides secondary indexing capabilities for Beanis documents
using native Redis data structures:
- Sets for categorical/string fields (exact match lookups)
- Sorted Sets for numeric fields (range queries)
"""

import json
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, field_validator

from beanis.odm.utils.pydantic import IS_PYDANTIC_V2

if IS_PYDANTIC_V2:
    from typing import Annotated
    from typing import get_args as typing_get_args

else:
    from typing_extensions import Annotated
    from typing_extensions import get_args as typing_get_args


class GeoPoint(BaseModel):
    """
    Represents a geographic point with longitude and latitude

    Usage:
        class Store(Document):
            name: str
            location: Indexed[GeoPoint]  # Geo index

        store = Store(name="HQ", location=GeoPoint(longitude=-122.4, latitude=37.8))
    """

    longitude: float
    latitude: float

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        return v

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v

    def as_tuple(self) -> Tuple[float, float]:
        """Return (longitude, latitude) tuple for Redis GEOADD"""
        return (self.longitude, self.latitude)


class IndexType:
    """Types of indexes supported"""

    SET = "set"  # For categorical/string fields (exact match)
    SORTED_SET = "zset"  # For numeric fields (range queries)
    GEO = "geo"  # For geo-spatial fields (location-based queries)


class IndexedField:
    """
    Marks a field as indexed for secondary index support

    Usage:
        class Product(Document):
            category: Annotated[str, IndexedField()]  # Set index
            price: Annotated[float, IndexedField()]   # Sorted Set index
            location: Annotated[GeoPoint, IndexedField()]  # Geo index
    """

    def __init__(self, index_type: Optional[str] = None):
        """
        :param index_type: Type of index ("set", "zset", or "geo").
                          If None, auto-detect based on field type
        """
        self.index_type = index_type

    def __repr__(self):
        return f"IndexedField(index_type={self.index_type})"


class IndexManager:
    """
    Manages secondary indexes for documents using Redis Sets and Sorted Sets
    """

    @staticmethod
    def get_index_key(
        document_class: Type, field_name: str, value: Any = None
    ) -> str:
        """
        Generate Redis key for an index

        For Set indexes: idx:Product:category:electronics
        For Sorted Set indexes: idx:Product:price
        """
        class_name = document_class.__name__

        if value is not None:
            # Set index (categorical)
            return f"idx:{class_name}:{field_name}:{value}"
        else:
            # Sorted Set index (numeric)
            return f"idx:{class_name}:{field_name}"

    @staticmethod
    def get_indexed_fields(document_class: Type) -> Dict[str, IndexedField]:
        """
        Extract all indexed fields from a document class

        Returns dict: {field_name: IndexedField}
        """
        indexed_fields = {}

        if IS_PYDANTIC_V2:
            # Pydantic v2: Check metadata
            for field_name, field_info in document_class.model_fields.items():
                if hasattr(field_info, "metadata") and field_info.metadata:
                    for metadata_item in field_info.metadata:
                        if isinstance(metadata_item, IndexedField):
                            indexed_fields[field_name] = metadata_item
                            break
        else:
            # Pydantic v1: Check outer_type_ for Annotated
            for field_name, field_info in document_class.__fields__.items():
                field_type = field_info.outer_type_
                if get_origin(field_type) is Annotated:
                    args = typing_get_args(field_type)
                    for arg in args[1:]:  # Skip first arg (the actual type)
                        if isinstance(arg, IndexedField):
                            indexed_fields[field_name] = arg
                            break

        return indexed_fields

    @staticmethod
    def determine_index_type(
        document_class: Type, field_name: str, indexed_field: IndexedField
    ) -> str:
        """
        Determine the index type based on field type

        - Numeric types (int, float) -> Sorted Set (zset)
        - GeoPoint types -> Geo index
        - String/categorical types -> Set
        """
        if indexed_field.index_type:
            return indexed_field.index_type

        # Get the field type from model
        if IS_PYDANTIC_V2:
            field_info = document_class.model_fields.get(field_name)
            if not field_info:
                return IndexType.SET
            field_type = field_info.annotation
        else:
            field_info = document_class.__fields__.get(field_name)
            if not field_info:
                return IndexType.SET
            field_type = field_info.outer_type_

        # Handle Annotated types
        if get_origin(field_type) is Annotated:
            args = get_args(field_type)
            field_type = args[0] if args else field_type

        # Handle Optional types
        if get_origin(field_type) is Union:
            args = get_args(field_type)
            # Get non-None type
            field_type = next(
                (arg for arg in args if arg is not type(None)), str
            )

        # Determine index type
        if field_type == GeoPoint or (
            isinstance(field_type, type) and issubclass(field_type, GeoPoint)
        ):
            return IndexType.GEO
        elif field_type in (int, float):
            return IndexType.SORTED_SET
        else:
            return IndexType.SET

    @staticmethod
    async def add_to_index(
        redis_client,
        document_class: Type,
        document_id: str,
        field_name: str,
        value: Any,
        index_type: str,
    ):
        """
        Add document ID to the appropriate index
        """
        if value is None:
            return  # Don't index None values

        if index_type == IndexType.SET:
            # Add to Set index
            index_key = IndexManager.get_index_key(
                document_class, field_name, value
            )
            await redis_client.sadd(index_key, document_id)

        elif index_type == IndexType.SORTED_SET:
            # Add to Sorted Set with value as score
            index_key = IndexManager.get_index_key(document_class, field_name)
            try:
                score = float(value)
                await redis_client.zadd(index_key, {document_id: score})
            except (ValueError, TypeError):
                # Can't convert to float, skip indexing
                pass

        elif index_type == IndexType.GEO:
            # Add to Geo index
            index_key = IndexManager.get_index_key(document_class, field_name)

            # Decode the value if it's in encoded format
            geo_value = value
            if isinstance(value, str) and value.startswith(
                "__type__:GeoPoint:"
            ):
                # Extract the JSON part
                encoded_json = value.split(":", 2)[2]
                geo_data = json.loads(encoded_json)
                geo_value = GeoPoint(**geo_data)

            if isinstance(geo_value, GeoPoint):
                await redis_client.geoadd(
                    index_key,
                    (geo_value.longitude, geo_value.latitude, document_id),
                )
            elif (
                isinstance(geo_value, dict)
                and "longitude" in geo_value
                and "latitude" in geo_value
            ):
                # Handle dict representation
                await redis_client.geoadd(
                    index_key,
                    (
                        geo_value["longitude"],
                        geo_value["latitude"],
                        document_id,
                    ),
                )

    @staticmethod
    async def remove_from_index(
        redis_client,
        document_class: Type,
        document_id: str,
        field_name: str,
        value: Any,
        index_type: str,
    ):
        """
        Remove document ID from the appropriate index
        """
        if value is None:
            return

        if index_type == IndexType.SET:
            # Remove from Set index
            index_key = IndexManager.get_index_key(
                document_class, field_name, value
            )
            await redis_client.srem(index_key, document_id)

        elif index_type == IndexType.SORTED_SET:
            # Remove from Sorted Set
            index_key = IndexManager.get_index_key(document_class, field_name)
            await redis_client.zrem(index_key, document_id)

        elif index_type == IndexType.GEO:
            # Remove from Geo index
            index_key = IndexManager.get_index_key(document_class, field_name)
            await redis_client.zrem(
                index_key, document_id
            )  # GEOADD uses sorted sets internally

    @staticmethod
    async def update_indexes(
        redis_client,
        document_class: Type,
        document_id: str,
        old_values: Optional[Dict[str, Any]],
        new_values: Dict[str, Any],
    ):
        """
        Update all indexes when a document changes
        Uses Redis pipeline for batch operations (performance optimization)

        :param old_values: Previous field values (for removal from old indexes)
        :param new_values: New field values (for adding to new indexes)
        """
        indexed_fields = IndexManager.get_indexed_fields(document_class)

        # Use pipeline for batch operations
        pipe = redis_client.pipeline()

        for field_name, indexed_field in indexed_fields.items():
            index_type = IndexManager.determine_index_type(
                document_class, field_name, indexed_field
            )

            old_value = old_values.get(field_name) if old_values else None
            new_value = new_values.get(field_name)

            # Remove from old index if value changed
            if old_value is not None and old_value != new_value:
                if index_type == IndexType.SET:
                    index_key = IndexManager.get_index_key(
                        document_class, field_name, old_value
                    )
                    pipe.srem(index_key, document_id)
                elif index_type == IndexType.SORTED_SET:
                    index_key = IndexManager.get_index_key(
                        document_class, field_name
                    )
                    pipe.zrem(index_key, document_id)
                elif index_type == IndexType.GEO:
                    index_key = IndexManager.get_index_key(
                        document_class, field_name
                    )
                    pipe.zrem(index_key, document_id)

            # Add to new index
            if new_value is not None:
                if index_type == IndexType.SET:
                    index_key = IndexManager.get_index_key(
                        document_class, field_name, new_value
                    )
                    pipe.sadd(index_key, document_id)
                elif index_type == IndexType.SORTED_SET:
                    index_key = IndexManager.get_index_key(
                        document_class, field_name
                    )
                    try:
                        score = float(new_value)
                        pipe.zadd(index_key, {document_id: score})
                    except (ValueError, TypeError):
                        pass  # Skip if can't convert to float
                elif index_type == IndexType.GEO:
                    index_key = IndexManager.get_index_key(
                        document_class, field_name
                    )

                    # Decode the value if it's in encoded format
                    geo_value = new_value
                    if isinstance(new_value, str) and new_value.startswith(
                        "__type__:GeoPoint:"
                    ):
                        # Extract the JSON part
                        encoded_json = new_value.split(":", 2)[2]
                        geo_data = json.loads(encoded_json)
                        geo_value = GeoPoint(**geo_data)

                    if isinstance(geo_value, GeoPoint):
                        pipe.geoadd(
                            index_key,
                            (
                                geo_value.longitude,
                                geo_value.latitude,
                                document_id,
                            ),
                        )
                    elif (
                        isinstance(geo_value, dict)
                        and "longitude" in geo_value
                        and "latitude" in geo_value
                    ):
                        pipe.geoadd(
                            index_key,
                            (
                                geo_value["longitude"],
                                geo_value["latitude"],
                                document_id,
                            ),
                        )

        # Execute all operations in a single round-trip
        await pipe.execute()

    @staticmethod
    async def remove_all_indexes(
        redis_client,
        document_class: Type,
        document_id: str,
        values: Dict[str, Any],
    ):
        """
        Remove document from all indexes (for deletion)
        Uses Redis pipeline for batch operations (performance optimization)
        """
        indexed_fields = IndexManager.get_indexed_fields(document_class)

        # Use pipeline for batch operations
        pipe = redis_client.pipeline()

        for field_name, indexed_field in indexed_fields.items():
            index_type = IndexManager.determine_index_type(
                document_class, field_name, indexed_field
            )
            value = values.get(field_name)

            if value is None:
                continue

            if index_type == IndexType.SET:
                index_key = IndexManager.get_index_key(
                    document_class, field_name, value
                )
                pipe.srem(index_key, document_id)
            elif index_type == IndexType.SORTED_SET:
                index_key = IndexManager.get_index_key(
                    document_class, field_name
                )
                pipe.zrem(index_key, document_id)
            elif index_type == IndexType.GEO:
                index_key = IndexManager.get_index_key(
                    document_class, field_name
                )
                pipe.zrem(index_key, document_id)

        # Execute all operations in a single round-trip
        await pipe.execute()

    @staticmethod
    async def find_by_index(
        redis_client,
        document_class: Type,
        field_name: str,
        value: Any = None,
        min_value: Any = None,
        max_value: Any = None,
    ) -> List[str]:
        """
        Find document IDs using an index

        For Set indexes (categorical):
            find_by_index(redis, Product, "category", value="electronics")

        For Sorted Set indexes (numeric range):
            find_by_index(redis, Product, "price", min_value=10, max_value=100)
        """
        indexed_fields = IndexManager.get_indexed_fields(document_class)

        if field_name not in indexed_fields:
            raise ValueError(f"Field '{field_name}' is not indexed")

        indexed_field = indexed_fields[field_name]
        index_type = IndexManager.determine_index_type(
            document_class, field_name, indexed_field
        )

        if index_type == IndexType.SET:
            # Exact match using Set
            if value is None:
                raise ValueError("value is required for Set index queries")

            index_key = IndexManager.get_index_key(
                document_class, field_name, value
            )
            members = await redis_client.smembers(index_key)

            # Convert bytes to strings if needed
            return [m.decode() if isinstance(m, bytes) else m for m in members]

        elif index_type == IndexType.SORTED_SET:
            # Range query using Sorted Set
            index_key = IndexManager.get_index_key(document_class, field_name)

            # Use -inf and +inf as defaults
            min_score = (
                float(min_value) if min_value is not None else float("-inf")
            )
            max_score = (
                float(max_value) if max_value is not None else float("+inf")
            )

            members = await redis_client.zrangebyscore(
                index_key, min_score, max_score
            )

            # Convert bytes to strings if needed
            return [m.decode() if isinstance(m, bytes) else m for m in members]

        return []

    @staticmethod
    async def find_by_geo_radius(
        redis_client,
        document_class: Type,
        field_name: str,
        longitude: float,
        latitude: float,
        radius: float,
        unit: str = "km",
    ) -> List[str]:
        """
        Find document IDs within a radius of a geo location

        :param field_name: Name of the geo-indexed field
        :param longitude: Center point longitude
        :param latitude: Center point latitude
        :param radius: Search radius
        :param unit: Distance unit - 'm', 'km', 'mi', 'ft' (default: 'km')

        Usage:
            nearby = await IndexManager.find_by_geo_radius(
                redis_client, Store, "location",
                longitude=-122.4, latitude=37.8,
                radius=10, unit="km"
            )
        """
        indexed_fields = IndexManager.get_indexed_fields(document_class)

        if field_name not in indexed_fields:
            raise ValueError(f"Field '{field_name}' is not indexed")

        indexed_field = indexed_fields[field_name]
        index_type = IndexManager.determine_index_type(
            document_class, field_name, indexed_field
        )

        if index_type != IndexType.GEO:
            raise ValueError(f"Field '{field_name}' is not a geo index")

        index_key = IndexManager.get_index_key(document_class, field_name)

        # Use GEORADIUS command
        members = await redis_client.georadius(
            index_key, longitude, latitude, radius, unit=unit
        )

        # Convert bytes to strings if needed
        return [m.decode() if isinstance(m, bytes) else m for m in members]

    @staticmethod
    async def find_by_geo_radius_with_distance(
        redis_client,
        document_class: Type,
        field_name: str,
        longitude: float,
        latitude: float,
        radius: float,
        unit: str = "km",
    ) -> List[Tuple[str, float]]:
        """
        Find document IDs within a radius with their distances

        Returns list of (document_id, distance) tuples

        Usage:
            nearby = await IndexManager.find_by_geo_radius_with_distance(
                redis_client, Store, "location",
                longitude=-122.4, latitude=37.8,
                radius=10, unit="km"
            )
            for doc_id, distance in nearby:
                print(f"{doc_id}: {distance} km away")
        """
        indexed_fields = IndexManager.get_indexed_fields(document_class)

        if field_name not in indexed_fields:
            raise ValueError(f"Field '{field_name}' is not indexed")

        indexed_field = indexed_fields[field_name]
        index_type = IndexManager.determine_index_type(
            document_class, field_name, indexed_field
        )

        if index_type != IndexType.GEO:
            raise ValueError(f"Field '{field_name}' is not a geo index")

        index_key = IndexManager.get_index_key(document_class, field_name)

        # Use GEORADIUS command with WITHDIST
        members = await redis_client.georadius(
            index_key, longitude, latitude, radius, unit=unit, withdist=True
        )

        # Convert bytes to strings if needed
        return [
            (m[0].decode() if isinstance(m[0], bytes) else m[0], float(m[1]))
            for m in members
        ]


# Convenience type for indexed fields
def Indexed(field_type: Type, **kwargs) -> Type:
    """
    Helper function to create an indexed field

    Usage:
        class Product(Document):
            category: Indexed[str]  # Set index
            price: Indexed[float]   # Sorted Set index
    """
    return Annotated[field_type, IndexedField(**kwargs)]


# Register GeoPoint encoder/decoder
def _register_geopoint_encoder():
    """Register encoder/decoder for GeoPoint to handle Redis serialization"""
    try:
        from beanis.odm.custom_encoders import register_type

        def encode_geopoint(gp: GeoPoint) -> str:
            return json.dumps(
                {"longitude": gp.longitude, "latitude": gp.latitude}
            )

        def decode_geopoint(s: str) -> GeoPoint:
            data = json.loads(s)
            return GeoPoint(**data)

        register_type(
            GeoPoint, encoder=encode_geopoint, decoder=decode_geopoint
        )
    except ImportError:
        pass  # Custom encoders module not available yet


# Auto-register GeoPoint encoder on module import
_register_geopoint_encoder()
