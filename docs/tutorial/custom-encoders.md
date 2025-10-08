# Custom Encoders

Beanis provides a custom encoder/decoder system to serialize and deserialize complex Python types to and from Redis.

## Overview

Redis stores data as strings, but you often need to work with complex Python types like NumPy arrays, PyTorch tensors, or custom objects. Custom encoders solve this problem:

- **Encoder** - Converts Python object → Redis string
- **Decoder** - Converts Redis string → Python object

## Quick Start

```python
from beanis import Document, register_encoder, register_decoder
import numpy as np
import base64
import pickle

# Register encoder
@register_encoder(np.ndarray)
def encode_numpy(arr: np.ndarray) -> str:
    return base64.b64encode(pickle.dumps(arr)).decode('utf-8')

# Register decoder
@register_decoder(np.ndarray)
def decode_numpy(data: str) -> np.ndarray:
    return pickle.loads(base64.b64decode(data.encode('utf-8')))

# Use in document
class MLModel(Document):
    name: str
    weights: np.ndarray  # Automatically encoded/decoded

# Works seamlessly
model = MLModel(name="my_model", weights=np.array([1, 2, 3]))
await model.insert()  # Weights encoded to string

retrieved = await MLModel.get(model.id)
print(retrieved.weights)  # np.array([1, 2, 3]) - decoded!
```

## Why Custom Encoders?

### Default Serialization

Beanis uses `msgspec` for fast JSON serialization, which handles:
- Basic types: `str`, `int`, `float`, `bool`, `None`
- Collections: `list`, `dict`, `set`, `tuple`
- Pydantic models: Nested documents
- Standard types: `datetime`, `UUID`, `Decimal`

### When You Need Custom Encoders

Use custom encoders for:
- **Scientific libraries**: NumPy, PyTorch, TensorFlow
- **Binary data**: Images, audio, compressed data
- **Custom classes**: Your own Python classes
- **Specialized formats**: Protocol buffers, MessagePack

## Registration Methods

### Method 1: Decorator (Recommended)

```python
from beanis import register_encoder, register_decoder

@register_encoder(MyType)
def encode_my_type(obj: MyType) -> str:
    return str(obj)  # Convert to string

@register_decoder(MyType)
def decode_my_type(data: str) -> MyType:
    return MyType(data)  # Convert back
```

### Method 2: Function Call

```python
from beanis import register_type

register_type(
    MyType,
    encoder=lambda obj: str(obj),
    decoder=lambda data: MyType(data)
)
```

### Method 3: Registry Class

```python
from beanis import CustomEncoderRegistry

CustomEncoderRegistry.register_encoder(MyType, encode_func)
CustomEncoderRegistry.register_decoder(MyType, decode_func)

# Or both at once
CustomEncoderRegistry.register_pair(MyType, encode_func, decode_func)
```

## Common Use Cases

### NumPy Arrays

```python
import numpy as np
import base64
import pickle
from beanis import register_type

register_type(
    np.ndarray,
    encoder=lambda arr: base64.b64encode(pickle.dumps(arr)).decode('utf-8'),
    decoder=lambda s: pickle.loads(base64.b64decode(s.encode('utf-8')))
)

class DataScience(Document):
    experiment: str
    results: np.ndarray

# Usage
doc = DataScience(
    experiment="test_1",
    results=np.array([[1, 2], [3, 4]])
)
await doc.insert()
```

### PyTorch Tensors

```python
import torch
import base64
import pickle
from beanis import register_type

register_type(
    torch.Tensor,
    encoder=lambda tensor: base64.b64encode(pickle.dumps(tensor)).decode('utf-8'),
    decoder=lambda s: pickle.loads(base64.b64decode(s.encode('utf-8')))
)

class NeuralNet(Document):
    model_name: str
    weights: torch.Tensor

# Usage
weights = torch.randn(10, 10)
model = NeuralNet(model_name="v1", weights=weights)
await model.insert()
```

### Pandas DataFrames

```python
import pandas as pd
from beanis import register_type

register_type(
    pd.DataFrame,
    encoder=lambda df: df.to_json(orient='split'),
    decoder=lambda s: pd.read_json(s, orient='split')
)

class Analysis(Document):
    dataset_name: str
    data: pd.DataFrame

# Usage
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
analysis = Analysis(dataset_name="sales", data=df)
await analysis.insert()
```

### PIL Images

```python
from PIL import Image
import io
import base64
from beanis import register_type

def encode_image(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def decode_image(data: str) -> Image.Image:
    buffer = io.BytesIO(base64.b64decode(data.encode('utf-8')))
    return Image.open(buffer)

register_type(Image.Image, encoder=encode_image, decoder=decode_image)

class ImageDoc(Document):
    title: str
    image: Image.Image

# Usage
img = Image.new('RGB', (100, 100), color='red')
doc = ImageDoc(title="test", image=img)
await doc.insert()
```

### Custom Classes

```python
from dataclasses import dataclass
import json
from beanis import register_type

@dataclass
class Point:
    x: float
    y: float

    def to_dict(self):
        return {'x': self.x, 'y': self.y}

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

register_type(
    Point,
    encoder=lambda p: json.dumps(p.to_dict()),
    decoder=lambda s: Point.from_dict(json.loads(s))
)

class Location(Document):
    name: str
    coordinates: Point

# Usage
loc = Location(name="Home", coordinates=Point(37.7749, -122.4194))
await loc.insert()
```

## Auto-Registration

Beanis automatically registers encoders for NumPy and PyTorch if installed:

```python
# No need to register manually!
import numpy as np
from beanis import Document

class Model(Document):
    weights: np.ndarray  # Auto-registered

# Just works
model = Model(weights=np.array([1, 2, 3]))
await model.insert()
```

To disable auto-registration:

```python
# In beanis/odm/custom_encoders.py
_AUTO_REGISTER = False  # Set before importing
```

## Advanced Patterns

### Conditional Encoding

```python
from beanis import register_encoder, register_decoder

@register_encoder(np.ndarray)
def encode_numpy(arr: np.ndarray) -> str:
    # Use compression for large arrays
    if arr.size > 1000:
        import zlib
        compressed = zlib.compress(pickle.dumps(arr))
        return "compressed:" + base64.b64encode(compressed).decode('utf-8')
    else:
        return base64.b64encode(pickle.dumps(arr)).decode('utf-8')

@register_decoder(np.ndarray)
def decode_numpy(data: str) -> np.ndarray:
    if data.startswith("compressed:"):
        import zlib
        data = data[11:]  # Remove prefix
        compressed = base64.b64decode(data.encode('utf-8'))
        return pickle.loads(zlib.decompress(compressed))
    else:
        return pickle.loads(base64.b64decode(data.encode('utf-8')))
```

### Type Variants

```python
from typing import Union
from beanis import register_encoder, register_decoder
import numpy as np
import torch

# Can't register Union directly, so register each type
@register_encoder(np.ndarray)
def encode_numpy(arr: np.ndarray) -> str:
    return "numpy:" + base64.b64encode(pickle.dumps(arr)).decode('utf-8')

@register_encoder(torch.Tensor)
def encode_torch(tensor: torch.Tensor) -> str:
    return "torch:" + base64.b64encode(pickle.dumps(tensor)).decode('utf-8')

# Single decoder that handles both
@register_decoder(np.ndarray)
def decode_array(data: str) -> np.ndarray:
    if data.startswith("numpy:"):
        data = data[6:]
    return pickle.loads(base64.b64decode(data.encode('utf-8')))

@register_decoder(torch.Tensor)
def decode_tensor(data: str) -> torch.Tensor:
    if data.startswith("torch:"):
        data = data[6:]
    return pickle.loads(base64.b64decode(data.encode('utf-8')))
```

### Versioned Encoding

```python
@register_encoder(MyClass)
def encode_v2(obj: MyClass) -> str:
    data = {
        'version': 2,
        'data': obj.to_dict()
    }
    return json.dumps(data)

@register_decoder(MyClass)
def decode_versioned(data: str) -> MyClass:
    parsed = json.loads(data)
    version = parsed.get('version', 1)

    if version == 1:
        # Old format
        return MyClass.from_old_format(parsed)
    elif version == 2:
        # New format
        return MyClass.from_dict(parsed['data'])
    else:
        raise ValueError(f"Unknown version: {version}")
```

## Performance Considerations

### Encoding Speed

Different serialization methods have different speeds:

```python
# Fastest - Simple JSON
register_type(
    Point,
    encoder=lambda p: json.dumps([p.x, p.y]),
    decoder=lambda s: Point(*json.loads(s))
)

# Fast - msgspec (if available)
import msgspec
register_type(
    Point,
    encoder=lambda p: msgspec.json.encode([p.x, p.y]).decode(),
    decoder=lambda s: Point(*msgspec.json.decode(s.encode()))
)

# Slower - Pickle (but handles more types)
register_type(
    ComplexType,
    encoder=lambda obj: base64.b64encode(pickle.dumps(obj)).decode(),
    decoder=lambda s: pickle.loads(base64.b64decode(s.encode()))
)
```

### Storage Size

Compression can reduce storage:

```python
import zlib

@register_encoder(LargeObject)
def encode_compressed(obj: LargeObject) -> str:
    serialized = pickle.dumps(obj)
    compressed = zlib.compress(serialized, level=6)
    return base64.b64encode(compressed).decode('utf-8')

@register_decoder(LargeObject)
def decode_compressed(data: str) -> LargeObject:
    compressed = base64.b64decode(data.encode('utf-8'))
    serialized = zlib.decompress(compressed)
    return pickle.loads(serialized)
```

## Error Handling

```python
@register_encoder(MyType)
def safe_encode(obj: MyType) -> str:
    try:
        return json.dumps(obj.to_dict())
    except Exception as e:
        # Log error or use fallback
        print(f"Encoding error: {e}")
        return json.dumps({'error': str(e)})

@register_decoder(MyType)
def safe_decode(data: str) -> MyType:
    try:
        parsed = json.loads(data)
        if 'error' in parsed:
            raise ValueError(f"Encoded error: {parsed['error']}")
        return MyType.from_dict(parsed)
    except json.JSONDecodeError as e:
        # Handle corrupt data
        print(f"Decoding error: {e}")
        return MyType()  # Return default
```

## Testing Encoders

```python
import pytest
from beanis import register_type

def test_custom_encoder():
    # Register encoder
    register_type(
        Point,
        encoder=lambda p: f"{p.x},{p.y}",
        decoder=lambda s: Point(*map(float, s.split(',')))
    )

    # Test encoding
    p = Point(1.5, 2.5)
    encoded = encode_point(p)
    assert encoded == "1.5,2.5"

    # Test decoding
    decoded = decode_point(encoded)
    assert decoded.x == 1.5
    assert decoded.y == 2.5

    # Test round-trip with document
    class Doc(Document):
        location: Point

    doc = Doc(location=Point(3.0, 4.0))
    # Would need actual Redis for full test
```

## Debugging

Check registered encoders:

```python
from beanis import CustomEncoderRegistry

# Check if type has encoder
encoder = CustomEncoderRegistry.get_encoder(MyType)
print(f"Encoder: {encoder}")

# Check if type has decoder
decoder = CustomEncoderRegistry.get_decoder(MyType)
print(f"Decoder: {decoder}")

# Clear all (for testing)
CustomEncoderRegistry.clear()
```

## Limitations

### Cannot Index Custom Types

```python
class Model(Document):
    weights: np.ndarray  # Cannot be indexed
    # weights: Indexed(np.ndarray)  # ❌ Won't work

# Workaround: Extract searchable fields
class Model(Document):
    weights: np.ndarray
    weight_size: Indexed(int)  # Can be indexed

    @before_event(Insert)
    async def set_metadata(self):
        self.weight_size = self.weights.size
```

### Type Hints Required

```python
# Must specify type hint
class Doc(Document):
    data: np.ndarray  # ✅ Works
    # data = None  # ❌ Type hint required for encoder
```

## Best Practices

1. **Use standard formats** - JSON over pickle when possible
2. **Version your encoders** - Include version field for compatibility
3. **Handle errors gracefully** - Corrupt data shouldn't crash
4. **Test round-trips** - Ensure encode→decode recovers original
5. **Consider compression** - For large objects
6. **Document your encoders** - Explain format for future maintainers
7. **Benchmark performance** - Profile with realistic data

## Comparison with Pydantic Validators

| Feature | Custom Encoders | Pydantic Validators |
|---------|----------------|-------------------|
| Purpose | Serialize to Redis | Validate/transform on input |
| When runs | Save/load from DB | Model instantiation |
| Type support | Any Python type | Pydantic-compatible types |
| Performance | Encoder-dependent | Very fast |
| Use case | Complex types | Input validation |

## Next Steps

- [Defining Documents](defining-a-document.md) - Using custom types in documents
- [Indexes](indexes.md) - Index limitations with custom types
- [Event Hooks](actions.md) - Derive searchable fields from custom types
- [API Reference](../api/custom-encoders.md) - Full encoder API

## Examples Repository

Check `tests/` for more examples:
- `test_custom_encoders.py` - Encoder tests
- `test_numpy_integration.py` - NumPy examples
- `test_complex_types.py` - Advanced patterns
