from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class IndexedAnnotation:
    _indexed: Tuple[int, Dict[str, Any]]


class ExpressionField(str):
    """
    Simple field expression for Redis ODM
    Removed query operator support (use indexing instead)
    """

    def __getitem__(self, item):
        """
        Get sub field

        :param item: name of the subfield
        :return: ExpressionField
        """
        return ExpressionField(f"{self}.{item}")

    def __getattr__(self, item):
        """
        Get sub field

        :param item: name of the subfield
        :return: ExpressionField
        """
        return ExpressionField(f"{self}.{item}")

    def __hash__(self):
        return hash(str(self))
