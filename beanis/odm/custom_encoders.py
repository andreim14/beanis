"""
Custom encoder/decoder registration system for Beanis

Allows users to register custom serialization logic for any Python type.
"""

import importlib.util
from typing import Any, Callable, Dict, Optional, Tuple, Type

# Type aliases for clarity
EncoderFunc = Callable[[Any], str]  # Converts object to string for Redis
DecoderFunc = Callable[[str], Any]  # Converts string from Redis to object
EncoderPair = Tuple[EncoderFunc, Optional[DecoderFunc]]


class CustomEncoderRegistry:
    """
    Global registry for custom type encoders/decoders

    Example usage:
        from beanis import register_encoder
        import numpy as np
        import base64
        import pickle

        @register_encoder(np.ndarray)
        def encode_numpy(arr: np.ndarray) -> str:
            return base64.b64encode(pickle.dumps(arr)).decode('utf-8')

        @register_decoder(np.ndarray)
        def decode_numpy(data: str) -> np.ndarray:
            return pickle.loads(base64.b64decode(data))
    """

    _encoders: Dict[Type, EncoderFunc] = {}
    _decoders: Dict[Type, DecoderFunc] = {}

    @classmethod
    def register_encoder(cls, type_: Type, encoder: EncoderFunc) -> None:
        """
        Register an encoder for a specific type

        Args:
            type_: The Python type to encode
            encoder: Function that converts the type to a string
        """
        cls._encoders[type_] = encoder

    @classmethod
    def register_decoder(cls, type_: Type, decoder: DecoderFunc) -> None:
        """
        Register a decoder for a specific type

        Args:
            type_: The Python type to decode
            decoder: Function that converts a string to the type
        """
        cls._decoders[type_] = decoder

    @classmethod
    def register_pair(
        cls, type_: Type, encoder: EncoderFunc, decoder: DecoderFunc
    ) -> None:
        """
        Register both encoder and decoder for a type

        Args:
            type_: The Python type
            encoder: Function that converts the type to a string
            decoder: Function that converts a string to the type
        """
        cls._encoders[type_] = encoder
        cls._decoders[type_] = decoder

    @classmethod
    def get_encoder(cls, type_: Type) -> Optional[EncoderFunc]:
        """Get encoder for a type, or None if not registered"""
        return cls._encoders.get(type_)

    @classmethod
    def get_decoder(cls, type_: Type) -> Optional[DecoderFunc]:
        """Get decoder for a type, or None if not registered"""
        return cls._decoders.get(type_)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered encoders/decoders (mainly for testing)"""
        cls._encoders.clear()
        cls._decoders.clear()


# Convenient decorator functions
def register_encoder(type_: Type) -> Callable[[EncoderFunc], EncoderFunc]:
    """
    Decorator to register a custom encoder for a type

    Example:
        @register_encoder(np.ndarray)
        def encode_numpy(arr: np.ndarray) -> str:
            return base64.b64encode(pickle.dumps(arr)).decode('utf-8')
    """

    def decorator(func: EncoderFunc) -> EncoderFunc:
        CustomEncoderRegistry.register_encoder(type_, func)
        return func

    return decorator


def register_decoder(type_: Type) -> Callable[[DecoderFunc], DecoderFunc]:
    """
    Decorator to register a custom decoder for a type

    Example:
        @register_decoder(np.ndarray)
        def decode_numpy(data: str) -> np.ndarray:
            return pickle.loads(base64.b64decode(data))
    """

    def decorator(func: DecoderFunc) -> DecoderFunc:
        CustomEncoderRegistry.register_decoder(type_, func)
        return func

    return decorator


def register_type(
    type_: Type, encoder: EncoderFunc, decoder: DecoderFunc
) -> None:
    """
    Register both encoder and decoder for a type (non-decorator version)

    Example:
        register_type(
            np.ndarray,
            encoder=lambda arr: base64.b64encode(pickle.dumps(arr)).decode(),
            decoder=lambda s: pickle.loads(base64.b64decode(s))
        )
    """
    CustomEncoderRegistry.register_pair(type_, encoder, decoder)


# Optional: Auto-register common types if libraries are installed
def _auto_register_common_types() -> None:
    """
    Automatically register encoders for common libraries if installed
    This is optional and only registers if the library is available
    """
    # NumPy support
    if importlib.util.find_spec("numpy") is not None:
        try:
            import base64
            import pickle

            import numpy as np

            register_type(
                np.ndarray,
                encoder=lambda arr: base64.b64encode(pickle.dumps(arr)).decode(
                    "utf-8"
                ),
                decoder=lambda s: pickle.loads(
                    base64.b64decode(s.encode("utf-8"))
                ),
            )
        except Exception:
            pass  # Fail silently if numpy is broken

    # PyTorch support
    if importlib.util.find_spec("torch") is not None:
        try:
            import base64
            import pickle

            import torch

            register_type(
                torch.Tensor,
                encoder=lambda tensor: base64.b64encode(
                    pickle.dumps(tensor)
                ).decode("utf-8"),
                decoder=lambda s: pickle.loads(
                    base64.b64decode(s.encode("utf-8"))
                ),
            )
        except Exception:
            pass  # Fail silently if torch is broken


# Auto-register on module import (optional - can be disabled)
_AUTO_REGISTER = True  # Set to False to disable auto-registration

if _AUTO_REGISTER:
    _auto_register_common_types()
