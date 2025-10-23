import decimal

import pydantic
from typing_extensions import Annotated

# Simplified Decimal support for Redis (no BSON dependency)
DecimalAnnotation = Annotated[
    decimal.Decimal,
    pydantic.BeforeValidator(
        lambda v: (
            decimal.Decimal(str(v))
            if not isinstance(v, decimal.Decimal)
            else v
        )
    ),
]
