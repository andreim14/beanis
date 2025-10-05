<a id="beanis.odm.indexes"></a>

## beanis.odm.indexes

> Redis-based indexing system using Sets and Sorted Sets
> 
> This module provides secondary indexing capabilities for Beanis documents
> using native Redis data structures:
> - Sets for categorical/string fields (exact match lookups)
> - Sorted Sets for numeric fields (range queries)

<a id="beanis.odm.indexes.IndexType"></a>

### IndexType

```python
class IndexType()
```

> Types of indexes supported

<a id="beanis.odm.indexes.IndexType.SET"></a>

#### IndexType.SET

> For categorical/string fields (exact match)

<a id="beanis.odm.indexes.IndexType.SORTED_SET"></a>

#### IndexType.SORTED\_SET

> For numeric fields (range queries)

<a id="beanis.odm.indexes.IndexedField"></a>

### IndexedField

```python
class IndexedField()
```

> Marks a field as indexed for secondary index support
> 
> Usage:
>     class Product(Document):
>         category: Annotated[str, IndexedField()]  # Set index
>         price: Annotated[float, IndexedField()]   # Sorted Set index

<a id="beanis.odm.indexes.IndexedField.__init__"></a>

#### IndexedField.\_\_init\_\_

```python
def __init__(index_type: Optional[str] = None)
```

> **Arguments**:
> 
> - `index_type`: Type of index ("set" or "zset").
> If None, auto-detect based on field type

<a id="beanis.odm.indexes.IndexManager"></a>

### IndexManager

```python
class IndexManager()
```

> Manages secondary indexes for documents using Redis Sets and Sorted Sets

<a id="beanis.odm.indexes.IndexManager.get_index_key"></a>

#### IndexManager.get\_index\_key

```python
@staticmethod
def get_index_key(document_class: Type,
                  field_name: str,
                  value: Any = None) -> str
```

> Generate Redis key for an index
> 
> For Set indexes: idx:Product:category:electronics
> For Sorted Set indexes: idx:Product:price

<a id="beanis.odm.indexes.IndexManager.get_indexed_fields"></a>

#### IndexManager.get\_indexed\_fields

```python
@staticmethod
def get_indexed_fields(document_class: Type) -> Dict[str, IndexedField]
```

> Extract all indexed fields from a document class
> 
> Returns dict: {field_name: IndexedField}

<a id="beanis.odm.indexes.IndexManager.determine_index_type"></a>

#### IndexManager.determine\_index\_type

```python
@staticmethod
def determine_index_type(document_class: Type, field_name: str,
                         indexed_field: IndexedField) -> str
```

> Determine the index type based on field type
> 
> - Numeric types (int, float) -> Sorted Set (zset)
> - String/categorical types -> Set

<a id="beanis.odm.indexes.IndexManager.add_to_index"></a>

#### IndexManager.add\_to\_index

```python
@staticmethod
async def add_to_index(redis_client, document_class: Type, document_id: str,
                       field_name: str, value: Any, index_type: str)
```

> Add document ID to the appropriate index

<a id="beanis.odm.indexes.IndexManager.remove_from_index"></a>

#### IndexManager.remove\_from\_index

```python
@staticmethod
async def remove_from_index(redis_client, document_class: Type,
                            document_id: str, field_name: str, value: Any,
                            index_type: str)
```

> Remove document ID from the appropriate index

<a id="beanis.odm.indexes.IndexManager.update_indexes"></a>

#### IndexManager.update\_indexes

```python
@staticmethod
async def update_indexes(redis_client, document_class: Type, document_id: str,
                         old_values: Optional[Dict[str, Any]],
                         new_values: Dict[str, Any])
```

> Update all indexes when a document changes
> 
> Uses Redis pipeline for batch operations (performance optimization)
> 
> **Arguments**:
> 
> - `old_values`: Previous field values (for removal from old indexes)
> - `new_values`: New field values (for adding to new indexes)

<a id="beanis.odm.indexes.IndexManager.remove_all_indexes"></a>

#### IndexManager.remove\_all\_indexes

```python
@staticmethod
async def remove_all_indexes(redis_client, document_class: Type,
                             document_id: str, values: Dict[str, Any])
```

> Remove document from all indexes (for deletion)
> Uses Redis pipeline for batch operations (performance optimization)

<a id="beanis.odm.indexes.IndexManager.find_by_index"></a>

#### IndexManager.find\_by\_index

```python
@staticmethod
async def find_by_index(redis_client,
                        document_class: Type,
                        field_name: str,
                        value: Any = None,
                        min_value: Any = None,
                        max_value: Any = None) -> List[str]
```

> Find document IDs using an index
> 
> For Set indexes (categorical):
>     find_by_index(redis, Product, "category", value="electronics")
> 
> For Sorted Set indexes (numeric range):
>     find_by_index(redis, Product, "price", min_value=10, max_value=100)

<a id="beanis.odm.indexes.Indexed"></a>

#### Indexed

```python
def Indexed(field_type: Type, **kwargs) -> Type
```

> Helper function to create an indexed field
> 
> Usage:
>     class Product(Document):
>         category: Indexed[str]  # Set index
>         price: Indexed[float]   # Sorted Set index

