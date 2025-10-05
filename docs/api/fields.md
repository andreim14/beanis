<a id="beanis.odm.fields"></a>

## beanis.odm.fields

<a id="beanis.odm.fields.ExpressionField"></a>

### ExpressionField

```python
class ExpressionField(str)
```

> Simple field expression for Redis ODM
> Removed query operator support (use indexing instead)

<a id="beanis.odm.fields.ExpressionField.__getitem__"></a>

#### ExpressionField.\_\_getitem\_\_

```python
def __getitem__(item)
```

> Get sub field
> 
> **Arguments**:
> 
> - `item`: name of the subfield
> 
> **Returns**:
> 
> ExpressionField

<a id="beanis.odm.fields.ExpressionField.__getattr__"></a>

#### ExpressionField.\_\_getattr\_\_

```python
def __getattr__(item)
```

> Get sub field
> 
> **Arguments**:
> 
> - `item`: name of the subfield
> 
> **Returns**:
> 
> ExpressionField

