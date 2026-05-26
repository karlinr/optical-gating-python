from abc import ABC, abstractmethod

predictor_registry = {}

def register_predictor(name):
    def decorator(cls):
        predictor_registry[name] = cls
        return cls
    return decorator

class PhasePredictor(ABC):
    """Base class for phase predictors."""
    @abstractmethod
    def update_phase(self, current_phase, timestamp, **kwargs):
        pass

    @abstractmethod
    def predict_target_time(self, target_phase, **kwargs):
        pass