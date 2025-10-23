import dataclasses as dc
import datetime
import decimal
import enum
import ipaddress
import operator
import pathlib
import uuid
from enum import Enum
from typing import (
    Any,
    Callable,
    Container,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
)

#import bson  # Removed for Redis
import pydantic

import beanis
from beanis.odm.custom_encoders import CustomEncoderRegistry
from beanis.odm.utils.pydantic import IS_PYDANTIC_V2, get_model_fields

SingleArgCallable = Callable[[Any], Any]
DEFAULT_CUSTOM_ENCODERS: MutableMapping[type, SingleArgCallable] = {
    ipaddress.IPv4Address: str,
    ipaddress.IPv4Interface: str,
    ipaddress.IPv4Network: str,
    ipaddress.IPv6Address: str,
    ipaddress.IPv6Interface: str,
    ipaddress.IPv6Network: str,
    pathlib.PurePath: str,
    pydantic.SecretBytes: pydantic.SecretBytes.get_secret_value,
    pydantic.SecretStr: pydantic.SecretStr.get_secret_value,
    datetime.date: lambda d: d.isoformat(),  # Convert to ISO string
    datetime.time: lambda t: t.isoformat(),  # Convert to ISO string
    datetime.timedelta: lambda td: td.total_seconds(),  # Convert to seconds (float)
    datetime.datetime: lambda dt: dt.isoformat(),  # Convert to ISO string
    enum.Enum: operator.attrgetter("value"),
    decimal.Decimal: str,  # Convert Decimal to string to preserve precision
    uuid.UUID: str,  # Convert UUID to string
}
if IS_PYDANTIC_V2:
    from pydantic_core import Url

    DEFAULT_CUSTOM_ENCODERS[Url] = str

REDIS_SCALAR_TYPES = (
    type(None),
    str,
    int,
    float,
    bool,  # Added bool as scalar
)


@dc.dataclass
class Encoder:
    """
    BSON encoding class
    """

    exclude: Container[str] = frozenset()
    custom_encoders: Mapping[type, SingleArgCallable] = dc.field(
        default_factory=dict
    )
    to_db: bool = False
    keep_nulls: bool = True

    def _encode_document(self, obj: "beanis.Document") -> Mapping[str, Any]:
        obj.parse_store()
        settings = obj.get_settings()
        obj_dict = {}

        if obj._class_id:
            obj_dict[settings.class_id] = obj._class_id

        sub_encoder = Encoder(
            # don't propagate self.exclude to subdocuments
            custom_encoders=settings.bson_encoders,
            to_db=self.to_db,
            keep_nulls=self.keep_nulls,
        )
        for key, value in self._iter_model_items(obj):
            obj_dict[key] = sub_encoder.encode(value)
        return obj_dict

    def encode(self, obj: Any) -> Any:
        # 1. Check document-specific custom encoders first
        if self.custom_encoders:
            encoder = _get_encoder(obj, self.custom_encoders)
            if encoder is not None:
                return encoder(obj)

        # 2. Check global custom encoder registry (user-registered types)
        obj_type = type(obj)
        encoder_func = CustomEncoderRegistry.get_encoder(obj_type)
        if encoder_func is not None:
            # Store type metadata with encoded value so decoder knows what type to use
            # Format: "__type__:TypeName:encoded_value"
            encoded_value = encoder_func(obj)
            return f"__type__:{obj_type.__name__}:{encoded_value}"

        # 3. Check DEFAULT_CUSTOM_ENCODERS (before scalar check)
        # This is important for types like Enum that might inherit from scalars (str, int, etc.)
        encoder = _get_encoder(obj, DEFAULT_CUSTOM_ENCODERS)
        if encoder is not None:
            return encoder(obj)

        # 4. Check if it's a Redis scalar type
        if isinstance(obj, REDIS_SCALAR_TYPES):
            return obj

        if isinstance(obj, beanis.Document):
            return self._encode_document(obj)
        if IS_PYDANTIC_V2 and isinstance(obj, pydantic.RootModel):
            return self.encode(obj.root)
        if isinstance(obj, pydantic.BaseModel):
            items = self._iter_model_items(obj)
            return {key: self.encode(value) for key, value in items}
        if isinstance(obj, Mapping):
            return {
                key if isinstance(key, Enum) else str(key): self.encode(value)
                for key, value in obj.items()
            }
        if isinstance(obj, Iterable):
            return [self.encode(value) for value in obj]

        raise ValueError(f"Cannot encode {obj!r}")

    def _iter_model_items(
        self, obj: pydantic.BaseModel
    ) -> Iterable[Tuple[str, Any]]:
        exclude, keep_nulls = self.exclude, self.keep_nulls
        get_model_field = get_model_fields(obj).get
        for key, value in obj.__iter__():
            field_info = get_model_field(key)
            if field_info is not None:
                key = field_info.alias or key
            if key not in exclude and (value is not None or keep_nulls):
                yield key, value


def _get_encoder(
    obj: Any, custom_encoders: Mapping[type, SingleArgCallable]
) -> Optional[SingleArgCallable]:
    encoder = custom_encoders.get(type(obj))
    if encoder is not None:
        return encoder
    for cls, encoder in custom_encoders.items():
        if isinstance(obj, cls):
            return encoder
    return None