<a id="beanis.odm.documents"></a>

## beanis.odm.documents

<a id="beanis.odm.documents.Document"></a>

### Document

```python
class Document(LazyModel, SettersInterface, InheritanceInterface,
               OtherGettersInterface)
```

> Document Mapping class for Redis.
> 
> Uses Redis Hashes for storage by default, with support for
> secondary indexes, TTL, and batch operations.

<a id="beanis.odm.documents.Document.get"></a>

#### Document.get

```python
@classmethod
async def get(cls: Type["DocType"], document_id: Any) -> Optional["DocType"]
```

> Get document by id, returns None if document does not exist
> 
> **Arguments**:
> 
> - `document_id`: str - document id
> 
> **Returns**:
> 
> Union["Document", None]

<a id="beanis.odm.documents.Document.exists"></a>

#### Document.exists

```python
@classmethod
async def exists(cls: Type["DocType"], document_id: Any) -> bool
```

> Check if a document exists by ID
> 
> **Arguments**:
> 
> - `document_id`: str - document id
> 
> **Returns**:
> 
> bool

<a id="beanis.odm.documents.Document.insert"></a>

#### Document.insert

```python
async def insert(ttl: Optional[int] = None) -> DocType
```

> Insert the document (self) to Redis
> 
> **Arguments**:
> 
> - `ttl`: Optional[int] - TTL in seconds
> 
> **Returns**:
> 
> Document

<a id="beanis.odm.documents.Document.insert_one"></a>

#### Document.insert\_one

```python
@classmethod
async def insert_one(cls: Type[DocType],
                     document: DocType,
                     ttl: Optional[int] = None) -> Optional[DocType]
```

> Insert one document to Redis
> 
> **Arguments**:
> 
> - `document`: Document - document to insert
> - `ttl`: Optional[int] - TTL in seconds
> 
> **Returns**:
> 
> DocType

<a id="beanis.odm.documents.Document.insert_many"></a>

#### Document.insert\_many

```python
@classmethod
async def insert_many(cls: Type[DocType],
                      documents: Iterable[DocType],
                      ttl: Optional[int] = None) -> List[DocType]
```

> Insert many documents to Redis using pipeline
> 
> **Arguments**:
> 
> - `documents`: List["Document"] - documents to insert
> - `ttl`: Optional[int] - TTL in seconds for all documents
> 
> **Returns**:
> 
> List[DocType]

<a id="beanis.odm.documents.Document.get_many"></a>

#### Document.get\_many

```python
@classmethod
async def get_many(cls: Type[DocType],
                   document_ids: List[Any]) -> List[Optional[DocType]]
```

> Get many documents by IDs using pipeline
> 
> Optimized with msgspec (2x faster than orjson) and model_construct() (skip validation)
> 
> Performance: 3-4x faster than orjson + model_validate approach
> 
> **Arguments**:
> 
> - `document_ids`: List[str] - list of document IDs
> 
> **Returns**:
> 
> List[Optional[DocType]]

<a id="beanis.odm.documents.Document.save"></a>

#### Document.save

```python
def save() -> DocType
```

> Update an existing model in Redis or insert it if it does not yet exist.
> 
> **Returns**:
> 
> Document

<a id="beanis.odm.documents.Document.update"></a>

#### Document.update

```python
async def update(**fields) -> DocType
```

> Update specific fields of the document
> 
> **Arguments**:
> 
> - `fields`: Field names and values to update
> 
> **Returns**:
> 
> Document

<a id="beanis.odm.documents.Document.get_field"></a>

#### Document.get\_field

```python
async def get_field(field_name: str) -> Any
```

> Get a specific field value from Redis without loading the entire document
> 
> **Arguments**:
> 
> - `field_name`: Name of the field
> 
> **Returns**:
> 
> Field value

<a id="beanis.odm.documents.Document.set_field"></a>

#### Document.set\_field

```python
async def set_field(field_name: str, value: Any) -> None
```

> Set a specific field value in Redis
> 
> **Arguments**:
> 
> - `field_name`: Name of the field
> - `value`: Value to set

<a id="beanis.odm.documents.Document.increment_field"></a>

#### Document.increment\_field

```python
async def increment_field(field_name: str,
                          amount: Union[int, float] = 1) -> Union[int, float]
```

> Increment a numeric field atomically
> 
> **Arguments**:
> 
> - `field_name`: Name of the field
> - `amount`: Amount to increment by
> 
> **Returns**:
> 
> New value

<a id="beanis.odm.documents.Document.set_ttl"></a>

#### Document.set\_ttl

```python
async def set_ttl(seconds: int) -> bool
```

> Set TTL (time to live) for this document
> 
> **Arguments**:
> 
> - `seconds`: TTL in seconds
> 
> **Returns**:
> 
> bool - True if TTL was set

<a id="beanis.odm.documents.Document.get_ttl"></a>

#### Document.get\_ttl

```python
async def get_ttl() -> Optional[int]
```

> Get the remaining TTL for this document
> 
> **Returns**:
> 
> Optional[int] - TTL in seconds, -1 if no TTL, -2 if key doesn't exist

<a id="beanis.odm.documents.Document.persist"></a>

#### Document.persist

```python
async def persist() -> bool
```

> Remove TTL from this document (make it persistent)
> 
> **Returns**:
> 
> bool - True if TTL was removed

<a id="beanis.odm.documents.Document.delete_self"></a>

#### Document.delete\_self

```python
@wrap_with_actions(EventTypes.DELETE)
async def delete_self(skip_actions=None)
```

> Delete the document

<a id="beanis.odm.documents.Document.delete"></a>

#### Document.delete

```python
@classmethod
async def delete(cls, document_id)
```

> Delete a document by ID
> 
> **Arguments**:
> 
> - `document_id`: str - document id

<a id="beanis.odm.documents.Document.delete_many"></a>

#### Document.delete\_many

```python
@classmethod
async def delete_many(cls, document_ids: List[Any]) -> int
```

> Delete many documents by IDs
> 
> **Arguments**:
> 
> - `document_ids`: List[str] - list of document IDs
> 
> **Returns**:
> 
> int - number of documents deleted

<a id="beanis.odm.documents.Document.delete_all"></a>

#### Document.delete\_all

```python
@classmethod
async def delete_all(cls) -> int
```

> Delete all documents of this class
> 
> **Returns**:
> 
> int - number of documents deleted

<a id="beanis.odm.documents.Document.count"></a>

#### Document.count

```python
@classmethod
async def count(cls) -> int
```

> Count all documents of this class
> 
> **Returns**:
> 
> int - number of documents

<a id="beanis.odm.documents.Document.find"></a>

#### Document.find

```python
@classmethod
async def find(cls: Type[DocType], **filters) -> List[DocType]
```

> Find documents by indexed fields
> 
> Examples:
>     # Exact match on indexed field
>     products = await Product.find(category="electronics")
> 
>     # Range query on numeric indexed field
>     products = await Product.find(price__gte=10, price__lte=100)
> 
> **Arguments**:
> 
> - `filters`: Field filters (supports __gte, __lte for numeric fields)
> 
> **Returns**:
> 
> List[DocType]

<a id="beanis.odm.documents.Document.all"></a>

#### Document.all

```python
@classmethod
async def all(cls: Type[DocType],
              skip: int = 0,
              limit: Optional[int] = None,
              sort_desc: bool = False) -> List[DocType]
```

> Get all documents of this class
> 
> **Arguments**:
> 
> - `skip`: Number of documents to skip
> - `limit`: Maximum number of documents to return
> - `sort_desc`: Sort by insertion time descending
> 
> **Returns**:
> 
> List[DocType]

<a id="beanis.odm.documents.Document.use_state_management"></a>

#### Document.use\_state\_management

```python
@classmethod
def use_state_management(cls) -> bool
```

> Is state management turned on
> 
> **Returns**:
> 
> bool

<a id="beanis.odm.documents.Document.state_management_save_previous"></a>

#### Document.state\_management\_save\_previous

```python
@classmethod
def state_management_save_previous(cls) -> bool
```

> Should we save the previous state after a commit to database
> 
> **Returns**:
> 
> bool

<a id="beanis.odm.documents.Document.state_management_replace_objects"></a>

#### Document.state\_management\_replace\_objects

```python
@classmethod
def state_management_replace_objects(cls) -> bool
```

> Should objects be replaced when using state management
> 
> **Returns**:
> 
> bool

<a id="beanis.odm.documents.Document.get_saved_state"></a>

#### Document.get\_saved\_state

```python
def get_saved_state() -> Optional[Dict[str, Any]]
```

> Saved state getter. It is protected property.
> 
> **Returns**:
> 
> Optional[Dict[str, Any]] - saved state

<a id="beanis.odm.documents.Document.get_previous_saved_state"></a>

#### Document.get\_previous\_saved\_state

```python
def get_previous_saved_state() -> Optional[Dict[str, Any]]
```

> Previous state getter. It is a protected property.
> 
> **Returns**:
> 
> Optional[Dict[str, Any]] - previous state

<a id="beanis.odm.documents.Document.get_settings"></a>

#### Document.get\_settings

```python
@classmethod
def get_settings(cls) -> ItemSettings
```

> Get document settings, which was created on
> 
> the initialization step
> 
> **Returns**:
> 
> ItemSettings class

