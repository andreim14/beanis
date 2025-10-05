<a id="beanis.odm.actions"></a>

## beanis.odm.actions

<a id="beanis.odm.actions.ActionRegistry"></a>

### ActionRegistry

```python
class ActionRegistry()
```

<a id="beanis.odm.actions.ActionRegistry.add_action"></a>

#### ActionRegistry.add\_action

```python
@classmethod
def add_action(cls, document_class: Type["Document"],
               event_types: List[EventTypes],
               action_direction: ActionDirections, funct: Callable)
```

> Add action to the action registry
> 
> **Arguments**:
> 
> - `document_class`: document class
> - `event_types`: List[EventTypes]
> - `action_direction`: ActionDirections - before or after
> - `funct`: Callable - function

<a id="beanis.odm.actions.ActionRegistry.get_action_list"></a>

#### ActionRegistry.get\_action\_list

```python
@classmethod
def get_action_list(cls, document_class: Type["Document"],
                    event_type: EventTypes,
                    action_direction: ActionDirections) -> List[Callable]
```

> Get stored action list
> 
> **Arguments**:
> 
> - `document_class`: Type - document class
> - `event_type`: EventTypes - type of needed event
> - `action_direction`: ActionDirections - before or after
> 
> **Returns**:
> 
> List[Callable] - list of stored methods

<a id="beanis.odm.actions.ActionRegistry.run_actions"></a>

#### ActionRegistry.run\_actions

```python
@classmethod
async def run_actions(cls, instance: "Document", event_type: EventTypes,
                      action_direction: ActionDirections,
                      exclude: List[Union[ActionDirections, str]])
```

> Run actions
> 
> **Arguments**:
> 
> - `instance`: Document - object of the Document subclass
> - `event_type`: EventTypes - event types
> - `action_direction`: ActionDirections - before or after

<a id="beanis.odm.actions.register_action"></a>

#### register\_action

```python
def register_action(event_types: Tuple[Union[List[EventTypes], EventTypes],
                                       ...],
                    action_direction: ActionDirections) -> Callable[[F], F]
```

> Decorator. Base registration method.
> 
> Used inside `before_event` and `after_event`
> 
> **Arguments**:
> 
> - `event_types`: Union[List[EventTypes], EventTypes] - event types
> - `action_direction`: ActionDirections - before or after

<a id="beanis.odm.actions.before_event"></a>

#### before\_event

```python
def before_event(
        *args: Union[List[EventTypes], EventTypes]) -> Callable[[F], F]
```

> Decorator. It adds action, which should run before mentioned one
> 
> or many events happen
> 
> **Arguments**:
> 
> - `args`: Union[List[EventTypes], EventTypes] - event types
> 
> **Returns**:
> 
> None

<a id="beanis.odm.actions.after_event"></a>

#### after\_event

```python
def after_event(
        *args: Union[List[EventTypes], EventTypes]) -> Callable[[F], F]
```

> Decorator. It adds action, which should run after mentioned one
> 
> or many events happen
> 
> **Arguments**:
> 
> - `args`: Union[List[EventTypes], EventTypes] - event types
> 
> **Returns**:
> 
> None

<a id="beanis.odm.actions.wrap_with_actions"></a>

#### wrap\_with\_actions

```python
def wrap_with_actions(
    event_type: EventTypes
) -> Callable[["AsyncDocMethod[DocType, P, R]"],
              "AsyncDocMethod[DocType, P, R]"]
```

> Helper function to wrap Document methods with
> 
> before and after event listeners
> 
> **Arguments**:
> 
> - `event_type`: EventTypes - event types
> 
> **Returns**:
> 
> None

