from beanis.odm.actions import (
    After,
    Before,
    Delete,
    Insert,
    Replace,
    Save,
    SaveChanges,
    Update,
    ValidateOnSave,
    after_event,
    before_event,
)
from beanis.odm.custom_types import DecimalAnnotation
from beanis.odm.documents import (
    Document,
    MergeStrategy,
)
from beanis.odm.indexes import IndexedField, Indexed

from beanis.odm.utils.init import init_beanis
from beanis.odm.custom_encoders import (
    register_encoder,
    register_decoder,
    register_type,
    CustomEncoderRegistry,
)

__version__ = "0.0.8"
__all__ = [
    # ODM
    "Document",
    "init_beanis",
    "MergeStrategy",
    # Indexes
    "IndexedField",
    "Indexed",
    # Actions
    "before_event",
    "after_event",
    "Insert",
    "Replace",
    "Save",
    "SaveChanges",
    "ValidateOnSave",
    "Delete",
    "Before",
    "After",
    "Update",
    # Custom Types
    "DecimalAnnotation",
    # Custom Encoders
    "register_encoder",
    "register_decoder",
    "register_type",
    "CustomEncoderRegistry",
]
