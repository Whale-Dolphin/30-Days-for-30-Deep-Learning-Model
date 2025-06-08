"""
Registry system for models and datasets with decorators.
"""

from typing import Dict, Type, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import BaseModel
    from .data import BaseDataset


class Registry:
    """Base registry class for registering components."""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: str = None):
        """Decorator for registering classes."""
        def decorator(cls):
            # Use class name if no name provided
            reg_name = name if name is not None else cls.__name__.lower()

            if reg_name in self._registry:
                raise ValueError(
                    f"{self.name} '{reg_name}' already registered")

            self._registry[reg_name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Type:
        """Get registered class by name."""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"{self.name} '{name}' not found. "
                f"Available: {available}"
            )
        return self._registry[name]

    def list_available(self) -> list:
        """List all available registered classes."""
        return list(self._registry.keys())

    def create(self, name: str, *args, **kwargs) -> Any:
        """Create instance of registered class."""
        cls = self.get(name)
        return cls(*args, **kwargs)


class DatasetRegistry(Registry):
    """Registry specifically for datasets with type classification."""

    def __init__(self):
        super().__init__("Dataset")
        self._iterable_datasets: Dict[str, Type] = {}
        self._map_style_datasets: Dict[str, Type] = {}

    def register(self, name: str = None, dataset_type: str = "map"):
        """
        Decorator for registering datasets.

        Args:
            name: Name to register under (default: class name lowercase)
            dataset_type: Type of dataset ("map" or "iterable")
        """
        def decorator(cls):
            # Import BaseDataset here to avoid circular import
            from .data import BaseDataset

            # Validate that it's a BaseDataset subclass
            if not issubclass(cls, BaseDataset):
                raise ValueError(
                    f"Class {cls.__name__} must inherit from BaseDataset")

            # Use class name if no name provided
            reg_name = name if name is not None else cls.__name__.lower()

            if reg_name in self._registry:
                raise ValueError(f"Dataset '{reg_name}' already registered")

            # Register in main registry
            self._registry[reg_name] = cls

            # Register in type-specific registry
            if dataset_type == "iterable":
                self._iterable_datasets[reg_name] = cls
            elif dataset_type == "map":
                self._map_style_datasets[reg_name] = cls
            else:
                raise ValueError(
                    f"Invalid dataset_type: {dataset_type}. Use 'map' or 'iterable'")

            # Store type info on class
            cls._dataset_type = dataset_type

            return cls
        return decorator

    def get_by_type(self, dataset_type: str) -> Dict[str, Type]:
        """Get all datasets of a specific type."""
        if dataset_type == "iterable":
            return self._iterable_datasets.copy()
        elif dataset_type == "map":
            return self._map_style_datasets.copy()
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")

    def list_by_type(self, dataset_type: str) -> list:
        """List available datasets of a specific type."""
        return list(self.get_by_type(dataset_type).keys())

    def get_dataset_type(self, name: str) -> str:
        """Get the type of a registered dataset."""
        cls = self.get(name)
        return getattr(cls, '_dataset_type', 'map')


class ModelRegistry(Registry):
    """Registry specifically for models."""

    def __init__(self):
        super().__init__("Model")

    def register(self, name: str = None):
        """Decorator for registering models."""
        def decorator(cls):
            # Import BaseModel here to avoid circular import
            from .models import BaseModel

            # Validate that it's a BaseModel subclass
            if not issubclass(cls, BaseModel):
                raise ValueError(
                    f"Class {cls.__name__} must inherit from BaseModel")

            # Use class name if no name provided
            reg_name = name if name is not None else cls.__name__.lower()

            if reg_name in self._registry:
                raise ValueError(f"Model '{reg_name}' already registered")

            self._registry[reg_name] = cls
            return cls
        return decorator


class PreprocessRegistry(Registry):
    """Registry specifically for preprocessing functions."""

    def __init__(self):
        super().__init__("Preprocess")

    def register(self, name: str = None):
        """Decorator for registering preprocessing functions."""
        def decorator(func):
            # Use function name if no name provided
            reg_name = name if name is not None else func.__name__.lower()

            if reg_name in self._registry:
                raise ValueError(f"Preprocess '{reg_name}' already registered")

            self._registry[reg_name] = func
            return func
        return decorator


# Global registries
MODELS = ModelRegistry()
DATASETS = DatasetRegistry()
PREPROCESSES = PreprocessRegistry()  # New global registry for preprocessing

# Convenience decorators


def register_model(name: str = None):
    """Decorator to register a model."""
    return MODELS.register(name)


def register_dataset(name: str = None, dataset_type: str = "map"):
    """Decorator to register a dataset."""
    return DATASETS.register(name, dataset_type)


def register_preprocess(name: str = None):
    """Decorator to register a preprocessing function."""
    return PREPROCESSES.register(name)


# Factory functions
def create_model(name: str, config: Dict[str, Any]):
    """Create a model instance from registry."""
    return MODELS.create(name, config)


def create_dataset(name: str, config: Dict[str, Any]):
    """Create a dataset instance from registry."""
    return DATASETS.create(name, config)


def list_models() -> list:
    """List all available models."""
    return MODELS.list_available()


def list_datasets(dataset_type: str = None) -> list:
    """List all available datasets, optionally filtered by type."""
    if dataset_type is None:
        return DATASETS.list_available()
    else:
        return DATASETS.list_by_type(dataset_type)
