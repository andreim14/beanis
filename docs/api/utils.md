<a id="beanis.odm.utils.init"></a>

## beanis.odm.utils.init

<a id="beanis.odm.utils.init.Initializer"></a>

### Initializer

```python
class Initializer()
```

<a id="beanis.odm.utils.init.Initializer.__init__"></a>

#### Initializer.\_\_init\_\_

```python
def __init__(database: "Redis" = None,
             document_models: Optional[List[Union[Type["DocType"],
                                                  str]]] = None)
```

> Beanis initializer
> 
> **Arguments**:
> 
> - `database`: redis.asyncio.Redis - Redis async client instance
> - `document_models`: List[Union[Type[DocType], str]] - model classes
> or strings with dot separated paths
> 
> **Returns**:
> 
> None

<a id="beanis.odm.utils.init.Initializer.get_model"></a>

#### Initializer.get\_model

```python
@staticmethod
def get_model(dot_path: str) -> Type["DocType"]
```

> Get the model by the path in format bar.foo.Model
> 
> **Arguments**:
> 
> - `dot_path`: str - dot seprated path to the model
> 
> **Returns**:
> 
> Type[DocType] - class of the model

<a id="beanis.odm.utils.init.Initializer.init_settings"></a>

#### Initializer.init\_settings

```python
def init_settings(cls: Union[Type[Document]])
```

> Init Settings
> 
> **Arguments**:
> 
> - `cls`: Union[Type[Document], Type[View], Type[UnionDoc]] - Class
> to init settings
> 
> **Returns**:
> 
> None

<a id="beanis.odm.utils.init.Initializer.set_default_class_vars"></a>

#### Initializer.set\_default\_class\_vars

```python
@staticmethod
def set_default_class_vars(cls: Type[Document])
```

> Set default class variables.
> 
> **Arguments**:
> 
> - `cls`: Union[Type[Document], Type[View], Type[UnionDoc]] - Class
> to init settings

<a id="beanis.odm.utils.init.Initializer.init_document_fields"></a>

#### Initializer.init\_document\_fields

```python
def init_document_fields(cls) -> None
```

> Init class fields
> 
> **Returns**:
> 
> None

<a id="beanis.odm.utils.init.Initializer.init_actions"></a>

#### Initializer.init\_actions

```python
@staticmethod
def init_actions(cls)
```

> Init event-based actions

<a id="beanis.odm.utils.init.Initializer.init_document_collection"></a>

#### Initializer.init\_document\_collection

```python
def init_document_collection(cls)
```

> Init Redis client for the Document-based class
> 
> **Arguments**:
> 
> - `cls`: 

<a id="beanis.odm.utils.init.Initializer.init_document"></a>

#### Initializer.init\_document

```python
async def init_document(cls: Type[Document]) -> Optional[Output]
```

> Init Document-based class
> 
> **Arguments**:
> 
> - `cls`: 

<a id="beanis.odm.utils.init.Initializer.init_class"></a>

#### Initializer.init\_class

```python
async def init_class(cls: Union[Type[Document]])
```

> Init Document, View or UnionDoc based class.
> 
> **Arguments**:
> 
> - `cls`: 

<a id="beanis.odm.utils.init.init_beanis"></a>

#### init\_beanis

```python
async def init_beanis(database: "Redis" = None,
                      document_models: Optional[List[Union[Type[Document],
                                                           str]]] = None)
```

> Beanis initialization
> 
> **Arguments**:
> 
> - `database`: redis.asyncio.Redis - Redis async client instance
> - `document_models`: List[Union[Type[DocType], str]] - model classes
> or strings with dot separated paths
> 
> **Returns**:
> 
> None

