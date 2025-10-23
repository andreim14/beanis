from typing import ClassVar, Optional

from beanis.odm.settings.document import DocumentSettings


class SettersInterface:
    _document_settings: ClassVar[Optional[DocumentSettings]]

    @classmethod
    def set_database(cls, database):
        """
        Redis client setter
        """
        cls._document_settings.redis_client = database

    @classmethod
    def set_collection_name(cls, name: str):
        """
        Key prefix setter (replaces collection name)
        """
        cls._document_settings.key_prefix = name
