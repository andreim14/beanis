

# Beanis - ODM de Redis para Humanos

[![Beanis](https://raw.githubusercontent.com/andreim14/beanis/main/assets/logo/logo-no-background.svg)](https://github.com/andreim14/beanis)

<div align="center">
  <a href="https://pypi.python.org/pypi/beanis"><img src="https://img.shields.io/pypi/v/beanis?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.python.org/pypi/beanis"><img src="https://img.shields.io/pypi/dm/beanis?color=blue" alt="Downloads"></a>
  <a href="https://pypi.python.org/pypi/beanis"><img src="https://img.shields.io/pypi/pyversions/beanis?color=blue" alt="Python versions"></a>
  <a href="https://github.com/andreim14/beanis/blob/main/LICENSE"><img src="https://img.shields.io/github/license/andreim14/beanis?color=blue" alt="License"></a>
  <a href="https://github.com/andreim14/beanis"><img src="https://img.shields.io/github/stars/andreim14/beanis?style=social" alt="GitHub stars"></a>
</div>

<div align="center">
  <a href="https://github.com/andreim14/beanis/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/andreim14/beanis/test.yml?branch=main&label=tests" alt="Tests"></a>
  <a href="https://github.com/andreim14/beanis"><img src="https://img.shields.io/badge/coverage-56%25-yellow" alt="Coverage"></a>
  <a href="https://github.com/andreim14/beanis"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  <a href="https://pydantic.dev"><img src="https://img.shields.io/badge/Pydantic-2.x-E92063?logo=pydantic" alt="Pydantic"></a>
  <a href="https://redis.io"><img src="https://img.shields.io/badge/Redis-7.x-DC382D?logo=redis&logoColor=white" alt="Redis"></a>
</div>

<div align="center">
  <b>
    📚 <a href="https://andreim14.github.io/beanis/">Documentación</a> |
    🚀 <a href="https://andreim14.github.io/beanis/getting-started/">Inicio Rápido</a> |
    📖 <a href="https://andreim14.github.io/beanis/tutorial/">Tutorial</a> |
    🔧 <a href="https://andreim14.github.io/beanis/api-documentation/document/">Referencia de la API</a>
  </b>
</div>

---

**Deja de escribir código boilerplate de Redis. Céntrate en la lógica de tu aplicación.**

Beanis es un ODM (Object-Document Mapper) asíncrono para Python en Redis que te ofrece modelos Pydantic, seguridad de tipos y una API limpia, mientras se mantiene rápido y funciona con Redis vanilla.

## ¿Por qué Beanis?

### El problema con Redis Vanilla

❌ **Serialización manual** - Escribes `json.dumps()` y `json.loads()` en todas partes
❌ **Conversiones de tipo** - Las cadenas de Redis necesitan conversiones manuales a `float()`, `int()`
❌ **Gestión de claves** - Administras manualmente claves como `"Product:123"`, `"all:Product"`
❌ **Sin validación** - Datos incorrectos corrupten silenciosamente tu base de datos en Redis
❌ **Código boilerplate** - 15-20 líneas para operaciones CRUD simples

### La solución: Beanis

✅ **Serialización automática** - Objetos anidados, listas, tipos personalizados, todo gestionado
✅ **Seguridad de tipos** - Validación completa de Pydantic + autocompletado de IDE
✅ **Gestión inteligente de claves** - Céntrate en tus datos, no en los internals de Redis
✅ **Validación de datos** - Detecta errores antes de que impacten en Redis
✅ **Escribe un 70% menos de código** - 5-7 líneas para las mismas operaciones

**Y es rápido:** Solo un 8% de sobrecarga frente a Redis vanilla

### ¿Quién debería usar Beanis?

✅ Estás construyendo una **aplicación de producción** que necesita Redis pero sin el código boilerplate
✅ Quieres **seguridad de tipos y validación** sin sacrificar rendimiento
✅ Estás usando **Redis vanilla** (sin módulos RedisJSON/RediSearch)
✅ Te gusta la **API de MongoDB de Beanie** y quieres lo mismo para Redis
✅ Almacenas **datos complejos** (objetos anidados, arrays de NumPy, etc.)

### ¿Cuándo NO usar Beanis?

❌ Necesitas cada microsegundo de rendimiento (usa redis-py nativo)
❌ Necesitas características de RedisJSON/RediSearch (usa Redis OM)
❌ Solo almacenas pares clave-valor simples (usa redis-py nativo)

## Muéstrame el Código

### Operación CRUD Básica

<table>
<tr>
<th>Redis Vanilla (20 líneas)</th>
<th>Beanis (7 líneas)</th>
</tr>
<tr>
<td>

```python
import json
import time
from redis.asyncio import Redis

redis = Redis(decode_responses=True)

# Insert
product_data = {
    "name": "Tony's Chocolonely",
    "price": "5.95",
    "category": json.dumps({
        "name": "Chocolate",
        "description": "Roasted cacao"
    })
}
await redis.hset("Product:prod_123",
                 mapping=product_data)
await redis.zadd("all:Product",
                 {"prod_123": time.time()})

# Retrieve
raw = await redis.hgetall("Product:prod_123")
product = {
    "name": raw["name"],
    "price": float(raw["price"]),
    "category": json.loads(raw["category"])
}
```

</td>
<td>

```python
from beanis import Document
from pydantic import BaseModel

class Category(BaseModel):
    name: str
    description: str

class Product(Document):
    name: str
    price: float
    category: Category

# Insert
product = Product(
    name="Tony's Chocolonely",
    price=5.95,
    category=Category(
        name="Chocolate",
        description="Roasted cacao"
    )
)
await product.insert()

# Retrieve
found = await Product.get(product.id)
```

</td>
</tr>
</table>

**Resultado:** Seguro de tipos, validado, **65% menos de código**

### Operación de Búsqueda/Consulta

<table>
<tr>
<th>Redis Vanilla (25 líneas)</th>
<th>Beanis (4 líneas)</th>
</tr>
<tr>
<td>

```python
# Find products between $10-50
keys = await redis.zrangebyscore(
    "idx:Product:price",
    min=10.0,
    max=50.0
)

# Fetch each product using pipeline
pipe = redis.pipeline()
for key in keys:
    pipe.hgetall(f"Product:{key}")
results = await pipe.execute()

# Parse manually
products = []
for data in results:
    if data:
        products.append({
            "name": data["name"],
            "price": float(data["price"]),
            "stock": int(data["stock"]),
            "category": json.loads(
                data.get("category", "{}")
            )
        })
```

</td>
<td>

```python
# Find products between $10-50
products = await Product.find(
    Product.price >= 10.0,
    Product.price <= 50.0
).to_list()
```

</td>
</tr>
</table>

**Resultado:** **84% menos de código**, resultados completamente tipados

### Operación de Actualización

<table>
<tr>
<th>Redis Vanilla (10 líneas)</th>
<th>Beanis (6 líneas)</th>
</tr>
<tr>
<td>

```python
# Update price and stock
await redis.hset("Product:123", mapping={
    "price": "6.95",
    "stock": "150"
})

# Atomic increment
new_stock = await redis.hincrby(
    "Product:123",
    "stock",
    -1
)
```

</td>
<td>

```python
# Update fields
await product.update(
    price=6.95,
    stock=150
)

# Atomic increment
new_stock = await product.increment_field(
    "stock", -1
)
```

</td>
</tr>
</table>

**Resultado:** Misma funcionalidad, API más limpia, seguro de tipos

### Operaciones por Lotes

<table>
<tr>
<th>Redis Vanilla (14 líneas)</th>
<th>Beanis (9 líneas)</th>
</tr>
<tr>
<td>

```python
# Insert 100 products
pipe = redis.pipeline()
for i in range(100):
    product_id = f"prod_{i}"
    data = {
        "name": f"Product {i}",
        "price": str(i * 10),
        "stock": "100",
        "category": json.dumps({
            "name": "Category"
        })
    }
    pipe.hset(f"Product:{product_id}",
              mapping=data)
    pipe.zadd("all:Product",
              {product_id: time.time()})
await pipe.execute()
```

</td>
<td>

```python
# Insert 100 products
products = [
    Product(
        name=f"Product {i}",
        price=i * 10,
        stock=100,
        category=Category(name="Category")
    )
    for i in range(100)
]
await Product.insert_many(products)
```

</td>
</tr>
</table>

**Resultado:** **35% menos de código**, sin gestión manual de claves

## Instalación

### PIP

```shell
pip install beanis
```

### Poetry

```shell
poetry add beanis
```

## Inicio Rápido

```python
import asyncio
from typing import Optional
from redis.asyncio import Redis
from pydantic import BaseModel
from beanis import Document, init_beanis, Indexed


class Category(BaseModel):
    name: str
    description: str


class Product(Document):
    name: str
    description: Optional[str] = None
    price: Indexed(float)  # Indexed for range queries
    category: Category
    stock: int = 0

    class Settings:
        name = "products"


async def main():
    # Initialize Redis client
    client = Redis(host="localhost", port=6379, db=0, decode_responses=True)

    # Initialize Beanis
    await init_beanis(database=client, document_models=[Product])

    # Create a product
    chocolate = Category(
        name="Chocolate",
        description="A preparation of roasted and ground cacao seeds."
    )

    product = Product(
        name="Tony's Chocolonely",
        price=5.95,
        category=chocolate,
        stock=100
    )

    # Insert into Redis
    await product.insert()

    # Retrieve by ID
    found = await Product.get(product.id)
    print(f"Found: {found.name} - ${found.price}")

    # Query by price range
    affordable = await Product.find(
        Product.price < 10.0
    ).to_list()
    print(f"Affordable products: {len(affordable)}")

    # Update specific fields
    await product.update(price=6.95, stock=150)

    # Atomic increment
    new_stock = await product.increment_field("stock", -1)
    print(f"Stock after sale: {new_stock}")

    # Get all products
    all_products = await Product.all()
    print(f"Total products: {len(all_products)}")

    # Delete
    await product.delete_self()

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## Características Principales

### 🚀 Seguridad de Tipos y Validación

Beanis utiliza modelos Pydantic, brindándote validación y verificación de tipos automáticas:

```python
class Product(Document):
    name: str
    price: float  # Must be a number
    stock: int    # Must be an integer
    category: Category  # Must be a valid Category object

# This will raise a validation error BEFORE hitting Redis
product = Product(
    name="Invalid",
    price="not a number",  # ❌ ValidationError!
    stock=100
)
```

### ⚡ Alto Rendimiento

Beanis está optimizado para velocidad con una sobrecarga mínima:

- **8% de sobrecarga** frente a Redis vanilla (benchmark realizado)
- Utiliza `msgspec` para un análisis JSON ultrarrápido (2x más rápido que orjson)
- Omite la validación de Pydantic en las lecturas por defecto (los datos de Redis se consideran confiables)
- Uso eficiente de pipelines para operaciones por lotes

**Resultados de Benchmark** (Obtener por ID):
- Redis Vanilla: 1.00x (línea base)
- Beanis: 1.08x (solo un 8% más lento)
- Redis OM: 1.20x (20% más lento)


### 🎯 API Pythonica

Interfaz estilo Beanie familiar para desarrolladores de MongoDB:

```python
# Query with Pythonic operators
products = await Product.find(
    Product.price >= 10.0,
    Product.price <= 50.0,
    Product.stock > 0
).to_list()

# Chaining operations
expensive = await Product.find(
    Product.price > 100
).sort(Product.price).limit(10).to_list()

# Batch operations
await Product.insert_many([product1, product2, product3])
products = await Product.get_many([id1, id2, id3])
await Product.delete_many([id1, id2])
```

### 📦 Almacena Cualquier Cosa

Beanis maneja tipos complejos automáticamente:

**Soporte nativo:**
- Modelos Pydantic anidados
- Listas, diccionarios, tuplas, conjuntos
- Decimal, UUID, Enum
- datetime, date, time, timedelta

**Tipos personalizados mediante codificadores:**
```python
from beanis import Document, register_type
import numpy as np

# NumPy arrays work automatically (auto-registered)
class MLModel(Document):
    name: str
    weights: Any  # Stores np.ndarray!

model = MLModel(name="v1", weights=np.random.rand(100, 100))
await model.insert()  # Just works!

# Custom types
register_type(
    MyCustomType,
    encoder=lambda obj: str(obj),
    decoder=lambda s: MyCustomType.from_string(s)
)
```

¡Consulta el tutorial para ver cómo crearlos!

### 🔧 Características Listas para Producción

**Soporte de TTL:**
```python
# Insert with TTL
await product.insert(ttl=3600)  # Expires in 1 hour

# Set TTL on existing document
await product.set_ttl(7200)
ttl = await product.get_ttl()
await product.persist()  # Remove TTL
```

**Ganchos de Eventos (Event Hooks):**
```python
from beanis import before_event, after_event, Insert, Update

class Product(Document):
    name: str
    price: float

    @before_event(Insert)
    async def validate_price(self):
        if self.price < 0:
            raise ValueError("Price cannot be negative")

    @after_event(Insert)
    async def log_creation(self):
        print(f"Created product: {self.name}")
```

**Operaciones a Nivel de Campo:**
```python
# Get/set single field without loading entire document
price = await product.get_field("price")
await product.set_field("stock", 200)

# Atomic increment
new_stock = await product.increment_field("stock", 5)
```

**Seguimiento de Documentos:**
```python
# Get all documents (sorted by insertion time)
all_products = await Product.all()

# Pagination
page1 = await Product.all(limit=10)
page2 = await Product.all(skip=10, limit=10)

# Count and delete all
count = await Product.count()
await Product.delete_all()
```

## Comparación

| Característica | Redis Vanilla | Beanis | Redis OM |
|---------|--------------|---------|----------|
| **Volumen de código** | 100% | **30%** ⭐ | 50% |
| **Seguridad de tipos** | Manual | **Automática** ⭐ | Automática |
| **Rendimiento** | **100%** ⭐ | 108% | 120% |
| **Redis Vanilla** | ✅ | **✅** ⭐ | ❌ Requiere módulos |
| **Validación** | Manual | **Automática** ⭐ | Automática |
| **Estilo de API** | Comandos de Redis | **Pythonico** ⭐ | Redis OM |
| **Curva de aprendizaje** | Media | **Fácil** ⭐ | Media |
| **Objetos anidados** | Manual | **Automática** ⭐ | Automática |
| **Tipos personalizados** | Manual | **Fácil** ⭐ | Limitado |
| **Ganchos de eventos** | ❌ | **✅** ⭐ | ❌ |
| **Todas las DB (0-15)** | ✅ | **✅** ⭐ | ❌ Solo DB 0 |

## Elije la Herramienta Correcta

### Elige Redis Vanilla cuando:
- Cada microsegundo importa (trading de alta frecuencia, etc.)
- Almacenamiento simple de pares clave-valor
- Eres un experto en Redis y no necesitas abstracciones

### Elige Beanis cuando: ⭐
- **Construyes aplicaciones de producción** con modelos de datos complejos
- Quieres **seguridad de tipos + rendimiento** (un 8% de sobrecarga es aceptable)
- Estás usando **Redis vanilla** (sin módulos RedisJSON/RediSearch)
- Necesitas almacenar **objetos anidados, tipos personalizados, arrays de NumPy**, etc.
- Vienes de **MongoDB/Beanie** y quieres una API familiar
- Necesitas **ganchos de eventos** para validación y gestión del ciclo de vida

### Elige Redis OM cuando:
- Necesitas características de **RedisJSON/RediSearch**
- No te importa instalar módulos de Redis
- Quieres integración con Redis Stack
- Necesitas búsqueda avanzada de texto completo

## Requisitos

- Python 3.8+
- Redis 5.0+
- Pydantic 1.10+ o 2.0+

## Pruebas

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=beanis

# Run specific test
pytest tests/test_core.py::test_insert_and_get
```

## Créditos

Beanis es un fork de [Beanie](https://github.com/BeanieODM/beanie), la increíble ODM para MongoDB creada por Roman Right y colaboradores.

Tomamos la base de código de Beanie y la reimaginamos por completo para Redis, reemplazando las operaciones de MongoDB con comandos de Redis mientras conservamos el elegante diseño de la API. Si usas MongoDB, echa un vistazo al [Beanie](https://github.com/BeanieODM/beanie) original: ¡es genial!

**Agradecimientos especiales a:**
- Roman Right y la comunidad de Beanie por crear los cimientos
- Todos los colaboradores de Beanie cuyo código inspiró este proyecto
- Los equipos de Redis y Pydantic por sus excelentes bibliotecas

[![Beanie](https://raw.githubusercontent.com/roman-right/beanie/main/assets/logo/white_bg.svg)](https://github.com/BeanieODM/beanie)

## Licencia

Licencia Apache 2.0
