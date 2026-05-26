from abc import ABC, abstractmethod

estimator_registry = {}

def register_estimator(name):
    def decorator(cls):
        estimator_registry[name] = cls
        return cls
    return decorator

class PhaseEstimator(ABC):
    dependencies = []

    def __init__(self):
        self._ready = False

    def is_ready(self):
        return self._ready

    @property
    def active_dependencies(self):
        """Returns the list of dependencies currently required by this estimator."""
        return self.dependencies

    @abstractmethod
    def update(self, frame, **kwargs):
        """Processes a frame. Returns an estimation dict if ready, or None if initializing."""
        pass