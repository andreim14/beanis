from abc import abstractmethod
from typing import TYPE_CHECKING

from beanis.odm.settings.base import ItemSettings

if TYPE_CHECKING:
    from redis.asyncio import Redis


class OtherGettersInterface:
    @classmethod
    @abstractmethod
    def get_settings(cls) -> ItemSettings:
        pass

    @classmethod
    def get_redis_client(cls) -> "Redis":
        """Get the Redis async client"""
        return cls.get_settings().redis_client

    @classmethod
    def get_collection_name(cls):
        """Get the key prefix (replaces collection name)"""
        return cls.get_settings().key_prefix or cls.__name__

    @classmethod
    def get_bson_encoders(cls):
        """Legacy method - kept for backward compatibility"""
        return cls.get_settings().bson_encoders

    @classmethod
    def get_link_fields(cls):
        """Legacy method - links not supported in Redis ODM"""
        return None
