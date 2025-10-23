from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ItemSettings(BaseModel):
    """
    Settings for Redis Document storage
    """
    name: Optional[str] = None
    class_id: str = "_class_id"
    is_root: bool = False

    # Redis client (using Any to avoid Pydantic validation issues)
    redis_client: Optional[Any] = None

    # Redis-specific settings
    key_prefix: Optional[str] = None  # e.g., "Product" -> "Product:id"
    use_hash: bool = True  # Use Redis Hash (vs JSON string)
    default_ttl: Optional[int] = None  # Default TTL in seconds

    # State management
    use_state_management: bool = False
    state_management_replace_objects: bool = False
    state_management_save_previous: bool = False

    # Validation
    validate_on_save: bool = False

    # Caching (for query results)
    use_cache: bool = False
    cache_expiration_time: int = 60
    cache_capacity: int = 32

    # Null handling
    keep_nulls: bool = True

    # For backward compatibility during transition
    bson_encoders: Dict[Any, Any] = Field(default_factory=dict)
    use_revision: bool = False

    class Config:
        arbitrary_types_allowed = True
