<a id="beanis.odm.interfaces.detector"></a>

## beanis.odm.interfaces.detector

<a id="beanis.odm.interfaces.getters"></a>

## beanis.odm.interfaces.getters

<a id="beanis.odm.interfaces.getters.OtherGettersInterface"></a>

### OtherGettersInterface

```python
class OtherGettersInterface()
```

<a id="beanis.odm.interfaces.getters.OtherGettersInterface.get_redis_client"></a>

#### OtherGettersInterface.get\_redis\_client

```python
@classmethod
def get_redis_client(cls) -> "Redis"
```

> Get the Redis async client

<a id="beanis.odm.interfaces.getters.OtherGettersInterface.get_collection_name"></a>

#### OtherGettersInterface.get\_collection\_name

```python
@classmethod
def get_collection_name(cls)
```

> Get the key prefix (replaces collection name)

<a id="beanis.odm.interfaces.getters.OtherGettersInterface.get_bson_encoders"></a>

#### OtherGettersInterface.get\_bson\_encoders

```python
@classmethod
def get_bson_encoders(cls)
```

> Legacy method - kept for backward compatibility

<a id="beanis.odm.interfaces.getters.OtherGettersInterface.get_link_fields"></a>

#### OtherGettersInterface.get\_link\_fields

```python
@classmethod
def get_link_fields(cls)
```

> Legacy method - links not supported in Redis ODM

<a id="beanis.odm.interfaces.setters"></a>

## beanis.odm.interfaces.setters

<a id="beanis.odm.interfaces.setters.SettersInterface"></a>

### SettersInterface

```python
class SettersInterface()
```

<a id="beanis.odm.interfaces.setters.SettersInterface.set_database"></a>

#### SettersInterface.set\_database

```python
@classmethod
def set_database(cls, database)
```

> Redis client setter

<a id="beanis.odm.interfaces.setters.SettersInterface.set_collection_name"></a>

#### SettersInterface.set\_collection\_name

```python
@classmethod
def set_collection_name(cls, name: str)
```

> Key prefix setter (replaces collection name)

<a id="beanis.odm.interfaces.clone"></a>

## beanis.odm.interfaces.clone

<a id="beanis.odm.interfaces.inheritance"></a>

## beanis.odm.interfaces.inheritance

