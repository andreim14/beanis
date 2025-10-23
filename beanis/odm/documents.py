import importlib
import inspect
import time
import uuid
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Coroutine,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

from lazy_model import LazyModel
from pydantic import (
    ConfigDict,
    Field,
)
from pydantic.main import BaseModel
from typing_extensions import Concatenate, ParamSpec, TypeAlias

from beanis.exceptions import (
    CollectionWasNotInitialized,
)
from beanis.odm.actions import (
    EventTypes,
    wrap_with_actions,
)
from beanis.odm.interfaces.detector import ModelType
from beanis.odm.interfaces.getters import OtherGettersInterface
from beanis.odm.interfaces.inheritance import InheritanceInterface
from beanis.odm.interfaces.setters import SettersInterface
from beanis.odm.settings.base import ItemSettings
from beanis.odm.utils.dump import get_dict
from beanis.odm.utils.parsing import merge_models, parse_obj
from beanis.odm.utils.pydantic import (
    IS_PYDANTIC_V2,
    get_extra_field_info,
    get_model_dump,
    get_model_fields,
    parse_model,
)
from beanis.odm.utils.state import (
    previous_saved_state_needed,
    saved_state_needed,
)
from beanis.odm.indexes import IndexManager

if IS_PYDANTIC_V2:
    pass

if TYPE_CHECKING:
    pass

FindType = TypeVar("FindType", bound=Union["Document", "View"])
DocType = TypeVar("DocType", bound="Document")
P = ParamSpec("P")
R = TypeVar("R")
AnyDocMethod: TypeAlias = Callable[Concatenate[DocType, P], R]
AsyncDocMethod: TypeAlias = Callable[
    Concatenate[DocType, P], Coroutine[Any, Any, R]
]
DocumentProjectionType = TypeVar("DocumentProjectionType", bound=BaseModel)


def json_schema_extra(schema: Dict[str, Any], model: Type["Document"]) -> None:
    # remove excluded fields from the json schema
    properties = schema.get("properties")
    if not properties:
        return
    for k, field in get_model_fields(model).items():
        k = field.alias or k
        if k not in properties:
            continue
        field_info = field if IS_PYDANTIC_V2 else field.field_info
        if field_info.exclude:
            del properties[k]


class MergeStrategy(str, Enum):
    local = "local"
    remote = "remote"


class Document(
    LazyModel,
    SettersInterface,
    InheritanceInterface,
    OtherGettersInterface,
):
    """
    Document Mapping class for Redis.

    Uses Redis Hashes for storage by default, with support for
    secondary indexes, TTL, and batch operations.
    """

    if IS_PYDANTIC_V2:
        model_config = ConfigDict(
            json_schema_extra=json_schema_extra,
            populate_by_name=True,
        )
    else:

        class Config:
            json_encoders = {}
            allow_population_by_field_name = True
            fields = {"id": "_id"}
            schema_extra = staticmethod(json_schema_extra)

    id: Optional[str] = Field(default=None, description="Document id")
    # Settings
    _document_settings: ClassVar[Optional[ItemSettings]] = None

    def __init__(self, *args, **kwargs) -> None:
        super(Document, self).__init__(*args, **kwargs)
        self.get_redis_client()

    def _get_redis_key(self) -> str:
        """Get the Redis key for this document"""
        settings = self.get_settings()
        prefix = settings.key_prefix or self.__class__.__name__
        return f"{prefix}:{self.id}"

    @classmethod
    def _get_redis_key_for_id(cls, document_id: str) -> str:
        """Get the Redis key for a given document ID"""
        settings = cls.get_settings()
        prefix = settings.key_prefix or cls.__name__
        return f"{prefix}:{document_id}"

    @classmethod
    def _get_tracking_key(cls) -> str:
        """Get the Redis key for the Sorted Set tracking all document IDs"""
        settings = cls.get_settings()
        prefix = settings.key_prefix or cls.__name__
        return f"all:{prefix}"

    @classmethod
    async def get(
        cls: Type["DocType"],
        document_id: Any,
    ) -> Optional["DocType"]:
        """
        Get document by id, returns None if document does not exist

        :param document_id: str - document id
        :return: Union["Document", None]
        """
        redis_client = cls.get_settings().redis_client
        redis_key = cls._get_redis_key_for_id(document_id)

        # Use HGETALL to get all fields from the Hash
        db_data = await redis_client.hgetall(redis_key)

        if not db_data:
            return None

        # Convert bytes keys/values to strings if needed
        if db_data and isinstance(next(iter(db_data.keys())), bytes):
            db_data = {
                k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
                for k, v in db_data.items()
            }

        # Add the ID to the data
        db_data["id"] = document_id

        # Get the class name to instantiate the correct type
        class_name = db_data.get("_class_name")
        if class_name:
            module_name, cls_name = class_name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls_loaded = getattr(module, cls_name)
            del db_data["_class_name"]
        else:
            cls_loaded = cls

        # Parse fields - convert empty strings back to None and parse JSON
        model_fields = get_model_fields(cls_loaded)
        from beanis.odm.custom_encoders import CustomEncoderRegistry

        for field_name in list(db_data.keys()):
            if field_name == "id" or field_name == "_class_name":
                continue

            field_value = db_data[field_name]

            # Empty string means None/null was stored
            if field_value == "":
                db_data[field_name] = None
                continue

            # Check for custom decoder first
            # Format: "__type__:TypeName:encoded_value"
            decoder_used = False
            if isinstance(field_value, str) and field_value.startswith("__type__:"):
                parts = field_value.split(":", 2)
                if len(parts) == 3:
                    type_name = parts[1]
                    encoded_value = parts[2]
                    # Find decoder by type name
                    found_decoder = False
                    for reg_type, decoder_func in CustomEncoderRegistry._decoders.items():
                        if reg_type.__name__ == type_name:
                            db_data[field_name] = decoder_func(encoded_value)
                            decoder_used = True
                            found_decoder = True
                            break
                    # If no decoder found, strip metadata and return encoded value
                    if not found_decoder:
                        db_data[field_name] = encoded_value
                        decoder_used = True

            # Try to parse as JSON for complex types
            if not decoder_used and field_name in model_fields:
                try:
                    import json
                    # Try parsing as JSON (for dicts, lists, nested objects)
                    parsed = json.loads(field_value)
                    db_data[field_name] = parsed
                except (json.JSONDecodeError, TypeError, ValueError):
                    # Not JSON, keep as is (string, number, etc.)
                    pass

        return cast(type(cls_loaded), parse_obj(cls_loaded, db_data))

    @classmethod
    async def exists(cls: Type["DocType"], document_id: Any) -> bool:
        """
        Check if a document exists by ID

        :param document_id: str - document id
        :return: bool
        """
        redis_client = cls.get_settings().redis_client
        redis_key = cls._get_redis_key_for_id(document_id)
        return await redis_client.exists(redis_key) > 0

    async def insert(self: DocType, ttl: Optional[int] = None) -> DocType:
        """
        Insert the document (self) to Redis
        :param ttl: Optional[int] - TTL in seconds
        :return: Document
        """
        return await Document.insert_one(self, ttl=ttl)

    @classmethod
    async def insert_one(
        cls: Type[DocType], document: DocType, ttl: Optional[int] = None
    ) -> Optional[DocType]:
        """
        Insert one document to Redis

        :param document: Document - document to insert
        :param ttl: Optional[int] - TTL in seconds
        :return: DocType
        """
        if not isinstance(document, cls):
            raise TypeError(
                "Inserting document must be of the original document class"
            )

        # Generate ID if not set
        if document.id is None:
            document.id = str(uuid.uuid4())

        redis_client = document.get_settings().redis_client
        redis_key = document._get_redis_key()

        # Convert document to dict for Hash storage
        to_save_dict = get_dict(
            document,
            to_db=True,
            keep_nulls=document.get_settings().keep_nulls,
        )

        # Remove id from dict (it's in the key)
        to_save_dict.pop("id", None)

        # Store class name for polymorphism support
        to_save_dict["_class_name"] = (
            document.__module__ + "." + document.__class__.__name__
        )

        # Flatten nested dicts to dot notation for Redis Hash
        flattened = {}
        for key, value in to_save_dict.items():
            if isinstance(value, dict):
                # Store nested objects as JSON strings
                import json
                flattened[key] = json.dumps(value)
            elif isinstance(value, (list, tuple)):
                # Store lists as JSON strings
                import json
                flattened[key] = json.dumps(value)
            else:
                flattened[key] = str(value) if value is not None else ""

        # Use pipeline for atomic operation
        async with redis_client.pipeline() as pipe:
            # Store document as Hash
            await pipe.hset(redis_key, mapping=flattened)

            # Set TTL if specified
            if ttl:
                await pipe.expire(redis_key, ttl)
            elif document.get_settings().default_ttl:
                await pipe.expire(redis_key, document.get_settings().default_ttl)

            # Add to tracking Sorted Set (score = timestamp)
            tracking_key = document._get_tracking_key()
            await pipe.zadd(tracking_key, {document.id: time.time()})

            # Execute pipeline
            await pipe.execute()

        # Update indexes (use document.__class__ to get the actual class)
        await IndexManager.update_indexes(
            redis_client,
            document.__class__,
            document.id,
            None,  # No old values for insert
            to_save_dict
        )

        return document

    @classmethod
    async def insert_many(
        cls: Type[DocType],
        documents: Iterable[DocType],
        ttl: Optional[int] = None,
    ) -> List[DocType]:
        """
        Insert many documents to Redis using pipeline

        :param documents: List["Document"] - documents to insert
        :param ttl: Optional[int] - TTL in seconds for all documents
        :return: List[DocType]
        """
        document_list = list(documents)
        if not document_list:
            return []

        redis_client = cls.get_settings().redis_client

        async with redis_client.pipeline() as pipe:
            for document in document_list:
                if not isinstance(document, cls):
                    raise TypeError(
                        "All documents must be of the original document class"
                    )

                # Generate ID if not set
                if document.id is None:
                    document.id = str(uuid.uuid4())

                redis_key = document._get_redis_key()

                # Convert document to dict
                to_save_dict = get_dict(
                    document,
                    to_db=True,
                    keep_nulls=document.get_settings().keep_nulls,
                )
                to_save_dict.pop("id", None)
                to_save_dict["_class_name"] = (
                    document.__module__ + "." + document.__class__.__name__
                )

                # Flatten for Hash storage
                flattened = {}
                for key, value in to_save_dict.items():
                    if isinstance(value, (dict, list, tuple)):
                        import json
                        flattened[key] = json.dumps(value)
                    else:
                        flattened[key] = str(value) if value is not None else ""

                # Add operations to pipeline
                await pipe.hset(redis_key, mapping=flattened)

                if ttl:
                    await pipe.expire(redis_key, ttl)
                elif document.get_settings().default_ttl:
                    await pipe.expire(redis_key, document.get_settings().default_ttl)

                # Add to tracking set
                tracking_key = document._get_tracking_key()
                await pipe.zadd(tracking_key, {document.id: time.time()})

            # Execute all operations
            await pipe.execute()

        return document_list

    @classmethod
    def _get_complex_fields_cache(cls) -> set:
        """
        Get cached set of field names that need JSON parsing (complex types)
        Cached at class level for performance
        """
        if not hasattr(cls, '_complex_fields_cache'):
            import json as stdlib_json
            model_fields = get_model_fields(cls)
            complex_fields = set()

            for field_name, field_info in model_fields.items():
                # Get field type
                if IS_PYDANTIC_V2:
                    field_type = field_info.annotation
                else:
                    field_type = field_info.outer_type_

                # Check if it's a complex type (dict, list, BaseModel)
                from typing import get_origin, get_args, Union
                origin = get_origin(field_type)

                # Unwrap Optional (which is Union[X, None])
                if origin is Union:
                    args = get_args(field_type)
                    # Get the non-None type from Optional
                    non_none_types = [arg for arg in args if arg is not type(None)]
                    if non_none_types:
                        field_type = non_none_types[0]
                        origin = get_origin(field_type)

                # Complex types that need JSON parsing
                if origin in (dict, list, tuple, set, frozenset) or (
                    isinstance(field_type, type) and issubclass(field_type, BaseModel)
                ):
                    complex_fields.add(field_name)

            cls._complex_fields_cache = complex_fields

        return cls._complex_fields_cache

    @classmethod
    def _get_field_types_cache(cls) -> dict:
        """
        Get cached dict of field_name -> (field_type, is_optional) for fast type conversion
        Used when skip validation is enabled to manually convert types
        """
        if not hasattr(cls, '_field_types_cache'):
            model_fields = get_model_fields(cls)
            field_types = {}

            for field_name, field_info in model_fields.items():
                # Get field type
                if IS_PYDANTIC_V2:
                    field_type = field_info.annotation
                else:
                    field_type = field_info.outer_type_

                from typing import get_origin, get_args, Union
                origin = get_origin(field_type)
                is_optional = False

                # Unwrap Optional (Union[X, None])
                if origin is Union:
                    args = get_args(field_type)
                    non_none_types = [arg for arg in args if arg is not type(None)]
                    if non_none_types:
                        field_type = non_none_types[0]
                        is_optional = len(args) > len(non_none_types)

                field_types[field_name] = (field_type, is_optional)

            cls._field_types_cache = field_types

        return cls._field_types_cache

    @classmethod
    async def get_many(
        cls: Type[DocType], document_ids: List[Any]
    ) -> List[Optional[DocType]]:
        """
        Get many documents by IDs using pipeline
        Optimized with msgspec (2x faster than orjson) and model_construct() (skip validation)

        Performance: 3-4x faster than orjson + model_validate approach

        :param document_ids: List[str] - list of document IDs
        :return: List[Optional[DocType]]
        """
        if not document_ids:
            return []

        redis_client = cls.get_settings().redis_client

        # Use pipeline to fetch all documents (batch all HGETALLs in single round-trip)
        pipe = redis_client.pipeline()
        for doc_id in document_ids:
            redis_key = cls._get_redis_key_for_id(doc_id)
            pipe.hgetall(redis_key)  # No await! Batch the command

        results = await pipe.execute()  # Execute all at once

        # Get complex fields that need JSON parsing (cached at class level)
        complex_fields = cls._get_complex_fields_cache()

        # Parse results
        documents = []
        import msgspec  # Ultra-fast JSON parsing (2x faster than orjson, 10-40x faster than stdlib)

        # Check if validation should be skipped (default: False for max performance)
        # Use getattr with default to handle models using ItemSettings instead of DocumentSettings
        use_validation = getattr(cls.get_settings(), 'use_validation_on_fetch', False)

        # Cache field types outside loop for performance (only needed if not validating)
        field_types_cache = None if use_validation else cls._get_field_types_cache()

        for doc_id, db_data in zip(document_ids, results):
            if not db_data:
                documents.append(None)
                continue

            # Convert bytes to strings if needed (should be rare with decode_responses=True)
            if db_data and isinstance(next(iter(db_data.keys())), bytes):
                db_data = {
                    k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
                    for k, v in db_data.items()
                }

            db_data["id"] = doc_id

            # Handle polymorphism
            class_name = db_data.get("_class_name")
            if class_name:
                module_name, cls_name = class_name.rsplit(".", 1)
                module = importlib.import_module(module_name)
                cls_loaded = getattr(module, cls_name)
                del db_data["_class_name"]
            else:
                cls_loaded = cls

            # COMBINED: Smart field parsing + type conversion in single loop (performance!)
            for field_name, field_value in list(db_data.items()):
                if field_name in ("id", "_class_name"):
                    continue

                # Empty string means None/null was stored
                if field_value == "":
                    db_data[field_name] = None
                    continue

                # Check for custom decoder FIRST (works for all field types)
                # We need to check by stored value metadata, not field annotation type
                # because field might be annotated as Any but contain specific type
                decoder_used = False
                from beanis.odm.custom_encoders import CustomEncoderRegistry

                # Check if we stored type metadata with the value
                # Format: "__type__:actual_type_name:encoded_value"
                if isinstance(field_value, str) and field_value.startswith("__type__:"):
                    parts = field_value.split(":", 2)
                    if len(parts) == 3:
                        type_name = parts[1]
                        encoded_value = parts[2]
                        # Find decoder by type name
                        found_decoder = False
                        for reg_type, decoder_func in CustomEncoderRegistry._decoders.items():
                            if reg_type.__name__ == type_name:
                                db_data[field_name] = decoder_func(encoded_value)
                                decoder_used = True
                                found_decoder = True
                                break
                        # If no decoder found, strip metadata and return encoded value
                        if not found_decoder:
                            db_data[field_name] = encoded_value
                            decoder_used = True

                # Parse JSON for complex fields (dicts, lists, nested models)
                if not decoder_used and field_name in complex_fields:
                    try:
                        # Use msgspec for ultra-fast JSON parsing (2x faster than orjson)
                        parsed = msgspec.json.decode(field_value.encode() if isinstance(field_value, str) else field_value)
                        db_data[field_name] = parsed

                        # When using model_construct(), convert collections to correct types
                        # (model_construct doesn't do this conversion, but model_validate does)
                        if not use_validation and field_types_cache and field_name in field_types_cache:
                            from typing import get_origin
                            field_type, is_optional = field_types_cache[field_name]
                            origin = get_origin(field_type)
                            # Convert list→set or list→tuple if needed
                            if origin == set and isinstance(parsed, list):
                                db_data[field_name] = set(parsed)
                            elif origin == tuple and isinstance(parsed, list):
                                db_data[field_name] = tuple(parsed)
                    except (msgspec.DecodeError, TypeError, ValueError):
                        pass  # Not JSON, keep as is
                # Type conversion for simple types (int, float, bool) when skipping validation
                elif not decoder_used and not use_validation and field_types_cache and field_name in field_types_cache and isinstance(field_value, str):
                    field_type, is_optional = field_types_cache[field_name]
                    try:
                        if field_type == int:
                            db_data[field_name] = int(field_value)
                        elif field_type == float:
                            db_data[field_name] = float(field_value)
                        elif field_type == bool:
                            db_data[field_name] = field_value.lower() in ('true', '1', 'yes')
                    except (ValueError, AttributeError):
                        pass  # Keep original value if conversion fails

            # PERFORMANCE: Use model_construct() to skip validation (30-50% speedup)
            # Data from Redis is trusted (already validated on insert)
            if not use_validation:
                # Fast path: Skip validation entirely
                if IS_PYDANTIC_V2:
                    documents.append(cast(type(cls_loaded), cls_loaded.model_construct(**db_data)))
                else:
                    documents.append(cast(type(cls_loaded), cls_loaded.construct(**db_data)))
            else:
                # Safe path: Full validation (slower but safer)
                if IS_PYDANTIC_V2:
                    documents.append(cast(type(cls_loaded), cls_loaded.model_validate(db_data)))
                else:
                    documents.append(cast(type(cls_loaded), parse_obj(cls_loaded, db_data)))

        return documents

    def save(self: DocType) -> DocType:
        """
        Update an existing model in Redis or insert it if it does not yet exist.
        :return: Document
        """
        return self.insert()

    async def update(self: DocType, **fields) -> DocType:
        """
        Update specific fields of the document

        :param fields: Field names and values to update
        :return: Document
        """
        if not self.id:
            raise ValueError("Cannot update document without an ID")

        redis_client = self.get_settings().redis_client
        redis_key = self._get_redis_key()

        # Update fields in the instance
        for field_name, value in fields.items():
            setattr(self, field_name, value)

        # Update in Redis Hash
        flattened = {}
        for key, value in fields.items():
            if isinstance(value, (dict, list, tuple)):
                import json
                flattened[key] = json.dumps(value)
            else:
                flattened[key] = str(value) if value is not None else ""

        await redis_client.hset(redis_key, mapping=flattened)

        return self

    async def get_field(self, field_name: str) -> Any:
        """
        Get a specific field value from Redis without loading the entire document

        :param field_name: Name of the field
        :return: Field value
        """
        if not self.id:
            raise ValueError("Cannot get field from document without an ID")

        redis_client = self.get_settings().redis_client
        redis_key = self._get_redis_key()

        value = await redis_client.hget(redis_key, field_name)

        if value is None:
            return None

        # Decode if bytes
        if isinstance(value, bytes):
            value = value.decode()

        # Try to parse JSON for complex types
        try:
            import json
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    async def set_field(self, field_name: str, value: Any) -> None:
        """
        Set a specific field value in Redis

        :param field_name: Name of the field
        :param value: Value to set
        """
        if not self.id:
            raise ValueError("Cannot set field on document without an ID")

        redis_client = self.get_settings().redis_client
        redis_key = self._get_redis_key()

        # Convert value to string for Hash storage
        if isinstance(value, (dict, list, tuple)):
            import json
            str_value = json.dumps(value)
        else:
            str_value = str(value) if value is not None else ""

        await redis_client.hset(redis_key, field_name, str_value)

        # Update local instance
        setattr(self, field_name, value)

    async def increment_field(self, field_name: str, amount: Union[int, float] = 1) -> Union[int, float]:
        """
        Increment a numeric field atomically

        :param field_name: Name of the field
        :param amount: Amount to increment by
        :return: New value
        """
        if not self.id:
            raise ValueError("Cannot increment field on document without an ID")

        redis_client = self.get_settings().redis_client
        redis_key = self._get_redis_key()

        if isinstance(amount, float):
            new_value = await redis_client.hincrbyfloat(redis_key, field_name, amount)
        else:
            new_value = await redis_client.hincrby(redis_key, field_name, amount)

        # Update local instance
        setattr(self, field_name, new_value)

        return new_value

    async def set_ttl(self, seconds: int) -> bool:
        """
        Set TTL (time to live) for this document

        :param seconds: TTL in seconds
        :return: bool - True if TTL was set
        """
        if not self.id:
            raise ValueError("Cannot set TTL on document without an ID")

        redis_client = self.get_settings().redis_client
        redis_key = self._get_redis_key()

        return await redis_client.expire(redis_key, seconds)

    async def get_ttl(self) -> Optional[int]:
        """
        Get the remaining TTL for this document

        :return: Optional[int] - TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        if not self.id:
            raise ValueError("Cannot get TTL for document without an ID")

        redis_client = self.get_settings().redis_client
        redis_key = self._get_redis_key()

        ttl = await redis_client.ttl(redis_key)

        if ttl == -2:
            return None  # Key doesn't exist
        return ttl

    async def persist(self) -> bool:
        """
        Remove TTL from this document (make it persistent)

        :return: bool - True if TTL was removed
        """
        if not self.id:
            raise ValueError("Cannot persist document without an ID")

        redis_client = self.get_settings().redis_client
        redis_key = self._get_redis_key()

        return await redis_client.persist(redis_key)

    @wrap_with_actions(EventTypes.DELETE)
    async def delete_self(self, skip_actions=None):
        """
        Delete the document
        """
        await self.delete(self.id)

    @classmethod
    async def delete(cls, document_id):
        """
        Delete a document by ID

        :param document_id: str - document id
        """
        redis_client = cls.get_settings().redis_client
        redis_key = cls._get_redis_key_for_id(document_id)
        tracking_key = cls._get_tracking_key()

        # Get current values to remove from indexes
        db_data = await redis_client.hgetall(redis_key)
        if db_data:
            # Convert bytes to strings
            if db_data and isinstance(next(iter(db_data.keys())), bytes):
                db_data = {
                    k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
                    for k, v in db_data.items()
                }

            # Remove from all indexes
            await IndexManager.remove_all_indexes(
                redis_client, cls, document_id, db_data
            )

        async with redis_client.pipeline() as pipe:
            # Delete the document Hash
            await pipe.delete(redis_key)
            # Remove from tracking Sorted Set
            await pipe.zrem(tracking_key, document_id)
            await pipe.execute()

    @classmethod
    async def delete_many(cls, document_ids: List[Any]) -> int:
        """
        Delete many documents by IDs

        :param document_ids: List[str] - list of document IDs
        :return: int - number of documents deleted
        """
        if not document_ids:
            return 0

        redis_client = cls.get_settings().redis_client
        tracking_key = cls._get_tracking_key()

        async with redis_client.pipeline() as pipe:
            for doc_id in document_ids:
                redis_key = cls._get_redis_key_for_id(doc_id)
                await pipe.delete(redis_key)
                await pipe.zrem(tracking_key, doc_id)

            results = await pipe.execute()

        # Count how many keys were actually deleted
        # Results alternate between DEL (count) and ZREM (count)
        deleted_count = sum(1 for i, r in enumerate(results) if i % 2 == 0 and r > 0)
        return deleted_count

    @classmethod
    async def delete_all(cls) -> int:
        """
        Delete all documents of this class

        :return: int - number of documents deleted
        """
        redis_client = cls.get_settings().redis_client
        tracking_key = cls._get_tracking_key()

        # Get all document IDs from tracking set
        all_ids = await redis_client.zrange(tracking_key, 0, -1)

        if not all_ids:
            return 0

        # Convert bytes to strings if needed
        all_ids = [
            id.decode() if isinstance(id, bytes) else id for id in all_ids
        ]

        # Delete all documents
        count = await cls.delete_many(all_ids)

        # Clear the tracking set
        await redis_client.delete(tracking_key)

        return count

    @classmethod
    async def count(cls) -> int:
        """
        Count all documents of this class

        :return: int - number of documents
        """
        redis_client = cls.get_settings().redis_client
        tracking_key = cls._get_tracking_key()

        return await redis_client.zcard(tracking_key)

    @classmethod
    async def find(
        cls: Type[DocType],
        **filters
    ) -> List[DocType]:
        """
        Find documents by indexed fields

        Examples:
            # Exact match on indexed field
            products = await Product.find(category="electronics")

            # Range query on numeric indexed field
            products = await Product.find(price__gte=10, price__lte=100)

        :param filters: Field filters (supports __gte, __lte for numeric fields)
        :return: List[DocType]
        """
        if not filters:
            # No filters, return all
            return await cls.all()

        redis_client = cls.get_settings().redis_client
        matching_ids = None

        for key, value in filters.items():
            # Parse filter key
            if "__" in key:
                # Range query: price__gte=10, price__lte=100
                field_name, operator = key.rsplit("__", 1)

                if operator == "gte":
                    # Greater than or equal
                    ids = await IndexManager.find_by_index(
                        redis_client, cls, field_name, min_value=value
                    )
                elif operator == "lte":
                    # Less than or equal
                    ids = await IndexManager.find_by_index(
                        redis_client, cls, field_name, max_value=value
                    )
                elif operator == "gt":
                    # Greater than (exclusive)
                    ids = await IndexManager.find_by_index(
                        redis_client, cls, field_name,
                        min_value=f"({value}"  # Exclusive in Redis
                    )
                elif operator == "lt":
                    # Less than (exclusive)
                    ids = await IndexManager.find_by_index(
                        redis_client, cls, field_name,
                        max_value=f"({value}"  # Exclusive in Redis
                    )
                else:
                    raise ValueError(f"Unsupported operator: {operator}")
            else:
                # Exact match: category="electronics"
                field_name = key
                ids = await IndexManager.find_by_index(
                    redis_client, cls, field_name, value=value
                )

            ids_set = set(ids)

            # Intersect with previous results (AND logic)
            if matching_ids is None:
                matching_ids = ids_set
            else:
                matching_ids = matching_ids.intersection(ids_set)

        if not matching_ids:
            return []

        # Fetch all matching documents
        documents = await cls.get_many(list(matching_ids))
        return [doc for doc in documents if doc is not None]

    @classmethod
    async def all(
        cls: Type[DocType],
        skip: int = 0,
        limit: Optional[int] = None,
        sort_desc: bool = False,
    ) -> List[DocType]:
        """
        Get all documents of this class

        :param skip: Number of documents to skip
        :param limit: Maximum number of documents to return
        :param sort_desc: Sort by insertion time descending
        :return: List[DocType]
        """
        redis_client = cls.get_settings().redis_client
        tracking_key = cls._get_tracking_key()

        # Get IDs from Sorted Set
        if limit is None:
            end = -1
        else:
            end = skip + limit - 1

        if sort_desc:
            all_ids = await redis_client.zrevrange(tracking_key, skip, end)
        else:
            all_ids = await redis_client.zrange(tracking_key, skip, end)

        if not all_ids:
            return []

        # Convert bytes to strings if needed
        all_ids = [
            id.decode() if isinstance(id, bytes) else id for id in all_ids
        ]

        # Fetch all documents
        documents = await cls.get_many(all_ids)

        # Filter out None values
        return [doc for doc in documents if doc is not None]

    # State management

    @classmethod
    def use_state_management(cls) -> bool:
        """
        Is state management turned on
        :return: bool
        """
        return cls.get_settings().use_state_management

    @classmethod
    def state_management_save_previous(cls) -> bool:
        """
        Should we save the previous state after a commit to database
        :return: bool
        """
        return cls.get_settings().state_management_save_previous

    @classmethod
    def state_management_replace_objects(cls) -> bool:
        """
        Should objects be replaced when using state management
        :return: bool
        """
        return cls.get_settings().state_management_replace_objects

    def _save_state(self) -> None:
        """
        Save current document state. Internal method
        :return: None
        """
        if self.use_state_management() and self.id is not None:
            if self.state_management_save_previous():
                self._previous_saved_state = self._saved_state

            self._saved_state = get_dict(
                self,
                to_db=True,
                keep_nulls=self.get_settings().keep_nulls,
                exclude={"revision_id"},
            )

    def get_saved_state(self) -> Optional[Dict[str, Any]]:
        """
        Saved state getter. It is protected property.
        :return: Optional[Dict[str, Any]] - saved state
        """
        return self._saved_state

    def get_previous_saved_state(self) -> Optional[Dict[str, Any]]:
        """
        Previous state getter. It is a protected property.
        :return: Optional[Dict[str, Any]] - previous state
        """
        return self._previous_saved_state

    @property
    @saved_state_needed
    def is_changed(self) -> bool:
        if self._saved_state == get_dict(
            self,
            to_db=True,
            keep_nulls=self.get_settings().keep_nulls,
            exclude={"revision_id"},
        ):
            return False
        return True

    @property
    @saved_state_needed
    @previous_saved_state_needed
    def has_changed(self) -> bool:
        if (
            self._previous_saved_state is None
            or self._previous_saved_state == self._saved_state
        ):
            return False
        return True

    def _collect_updates(
        self, old_dict: Dict[str, Any], new_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compares old_dict with new_dict and returns field paths that have been updated
        Args:
            old_dict: dict1
            new_dict: dict2

        Returns: dictionary with updates

        """
        updates = {}
        if old_dict.keys() - new_dict.keys():
            updates = new_dict
        else:
            for field_name, field_value in new_dict.items():
                if field_value != old_dict.get(field_name):
                    if not self.state_management_replace_objects() and (
                        isinstance(field_value, dict)
                        and isinstance(old_dict.get(field_name), dict)
                    ):
                        if old_dict.get(field_name) is None:
                            updates[field_name] = field_value
                        elif isinstance(field_value, dict) and isinstance(
                            old_dict.get(field_name), dict
                        ):
                            field_data = self._collect_updates(
                                old_dict.get(field_name),  # type: ignore
                                field_value,
                            )

                            for k, v in field_data.items():
                                updates[f"{field_name}.{k}"] = v
                    else:
                        updates[field_name] = field_value

        return updates

    @saved_state_needed
    def get_changes(self) -> Dict[str, Any]:
        return self._collect_updates(
            self._saved_state,  # type: ignore
            get_dict(
                self,
                to_db=True,
                keep_nulls=self.get_settings().keep_nulls,
                exclude={"revision_id"},
            ),
        )

    @saved_state_needed
    @previous_saved_state_needed
    def get_previous_changes(self) -> Dict[str, Any]:
        if self._previous_saved_state is None:
            return {}

        return self._collect_updates(
            self._previous_saved_state,
            self._saved_state,  # type: ignore
        )

    @classmethod
    def get_settings(cls) -> ItemSettings:
        """
        Get document settings, which was created on
        the initialization step

        :return: ItemSettings class
        """
        if cls._document_settings is None:
            raise CollectionWasNotInitialized
        return cls._document_settings

    @classmethod
    def check_hidden_fields(cls):
        hidden_fields = [
            (name, field)
            for name, field in get_model_fields(cls).items()
            if get_extra_field_info(field, "hidden") is True
        ]
        if not hidden_fields:
            return
        import warnings
        warnings.warn(
            f"{cls.__name__}: 'hidden=True' is deprecated, please use 'exclude=True'",
            DeprecationWarning,
        )
        if IS_PYDANTIC_V2:
            for name, field in hidden_fields:
                field.exclude = True
                del field.json_schema_extra["hidden"]
            cls.model_rebuild(force=True)
        else:
            for name, field in hidden_fields:
                field.field_info.exclude = True
                del field.field_info.extra["hidden"]
                cls.__exclude_fields__[name] = True

    @wrap_with_actions(event_type=EventTypes.VALIDATE_ON_SAVE)
    async def validate_self(self, *args, **kwargs):
        # TODO: it can be sync, but needs some actions controller improvements
        if self.get_settings().validate_on_save:
            new_model = parse_model(self.__class__, get_model_dump(self))
            merge_models(self, new_model)

    @classmethod
    def get_model_type(cls) -> ModelType:
        return ModelType.Document
