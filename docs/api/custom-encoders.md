<a id="beanis.odm.custom_encoders"></a>

## beanis.odm.custom\_encoders

> Custom encoder/decoder registration system for Beanis
> 
> Allows users to register custom serialization logic for any Python type.

<a id="beanis.odm.custom_encoders.EncoderFunc"></a>

#### beanis.odm.custom\_encoders.EncoderFunc

> Converts object to string for Redis

<a id="beanis.odm.custom_encoders.DecoderFunc"></a>

#### beanis.odm.custom\_encoders.DecoderFunc

> Converts string from Redis to object

<a id="beanis.odm.custom_encoders.CustomEncoderRegistry"></a>

### CustomEncoderRegistry

```python
class CustomEncoderRegistry()
```

> Global registry for custom type encoders/decoders
> 
> Example usage:
>     from beanis import register_encoder
>     import numpy as np
>     import base64
>     import pickle
> 
>     @register_encoder(np.ndarray)
>     def encode_numpy(arr: np.ndarray) -> str:
>         return base64.b64encode(pickle.dumps(arr)).decode('utf-8')
> 
>     @register_decoder(np.ndarray)
>     def decode_numpy(data: str) -> np.ndarray:
>         return pickle.loads(base64.b64decode(data))

<a id="beanis.odm.custom_encoders.CustomEncoderRegistry.register_encoder"></a>

#### CustomEncoderRegistry.register\_encoder

```python
@classmethod
def register_encoder(cls, type_: Type, encoder: EncoderFunc) -> None
```

> Register an encoder for a specific type
> 
> **Arguments**:
> 
> - `type_` - The Python type to encode
> - `encoder` - Function that converts the type to a string

<a id="beanis.odm.custom_encoders.CustomEncoderRegistry.register_decoder"></a>

#### CustomEncoderRegistry.register\_decoder

```python
@classmethod
def register_decoder(cls, type_: Type, decoder: DecoderFunc) -> None
```

> Register a decoder for a specific type
> 
> **Arguments**:
> 
> - `type_` - The Python type to decode
> - `decoder` - Function that converts a string to the type

<a id="beanis.odm.custom_encoders.CustomEncoderRegistry.register_pair"></a>

#### CustomEncoderRegistry.register\_pair

```python
@classmethod
def register_pair(cls, type_: Type, encoder: EncoderFunc,
                  decoder: DecoderFunc) -> None
```

> Register both encoder and decoder for a type
> 
> **Arguments**:
> 
> - `type_` - The Python type
> - `encoder` - Function that converts the type to a string
> - `decoder` - Function that converts a string to the type

<a id="beanis.odm.custom_encoders.CustomEncoderRegistry.get_encoder"></a>

#### CustomEncoderRegistry.get\_encoder

```python
@classmethod
def get_encoder(cls, type_: Type) -> Optional[EncoderFunc]
```

> Get encoder for a type, or None if not registered

<a id="beanis.odm.custom_encoders.CustomEncoderRegistry.get_decoder"></a>

#### CustomEncoderRegistry.get\_decoder

```python
@classmethod
def get_decoder(cls, type_: Type) -> Optional[DecoderFunc]
```

> Get decoder for a type, or None if not registered

<a id="beanis.odm.custom_encoders.CustomEncoderRegistry.clear"></a>

#### CustomEncoderRegistry.clear

```python
@classmethod
def clear(cls) -> None
```

> Clear all registered encoders/decoders (mainly for testing)

<a id="beanis.odm.custom_encoders.register_encoder"></a>

#### register\_encoder

```python
def register_encoder(type_: Type) -> Callable[[EncoderFunc], EncoderFunc]
```

> Decorator to register a custom encoder for a type
> 
> **Example**:
> 
>   @register_encoder(np.ndarray)
>   def encode_numpy(arr: np.ndarray) -> str:
>   return base64.b64encode(pickle.dumps(arr)).decode('utf-8')

<a id="beanis.odm.custom_encoders.register_decoder"></a>

#### register\_decoder

```python
def register_decoder(type_: Type) -> Callable[[DecoderFunc], DecoderFunc]
```

> Decorator to register a custom decoder for a type
> 
> **Example**:
> 
>   @register_decoder(np.ndarray)
>   def decode_numpy(data: str) -> np.ndarray:
>   return pickle.loads(base64.b64decode(data))

<a id="beanis.odm.custom_encoders.register_type"></a>

#### register\_type

```python
def register_type(type_: Type, encoder: EncoderFunc,
                  decoder: DecoderFunc) -> None
```

> Register both encoder and decoder for a type (non-decorator version)
> 
> **Example**:
> 
>   register_type(
>   np.ndarray,
>   encoder=lambda arr: base64.b64encode(pickle.dumps(arr)).decode(),
>   decoder=lambda s: pickle.loads(base64.b64decode(s))
>   )

