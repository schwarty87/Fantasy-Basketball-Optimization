from typing import Dict, Type, Any, Callable


class Registry:
    """A generic registry to store and retrieve classes by name."""

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type[Any]] = {}

    def register(self, _cls: Type[Any] | None = None, *, name: str | None = None) -> Callable:
        """
        A decorator to register a class. It can be used in two ways:
        - As @registry_instance.register: The class name is used as the registration key
        - As @registry_instance.register(name="CustomName"): The provided name is used as the key
        """

        def decorator(cls: Type[Any]):
            registration_name = name if name is not None else cls.__name__
            if registration_name in self._registry:
                raise ValueError(f"Error in '{self._name}' registry: Name {registration_name} is already registered.")
            self._registry[registration_name] = cls
            return cls

        if _cls is None:
            return decorator  # used as @registry_instance.register(name=...)
        else:
            return decorator(_cls)  # used as @registry_instance.register

    def get(self, name: str) -> Type[Any]:
        if name not in self._registry:
            raise ValueError(f"Name '{name}' is not in the {self._name} registry.")
        return self._registry[name]

# Instantiate all the registries
callbacks_registry = Registry(name="Callback")
datasets_registry = Registry(name="Dataset")
enums_registry = Registry(name="Enum")
layers_registry = Registry(name="Layer")
metrics_registry = Registry(name="Metric")
params_registry = Registry(name="Params")

