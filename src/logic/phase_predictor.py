import numpy as np
from app.config import Config
from abc import ABC, abstractmethod

TWO_PI = 2 * np.pi

class PhasePredictor(ABC):
    """Base class for phase predictors."""
    @abstractmethod
    def update_phase(self, current_phase, timestamp):
        pass

    @abstractmethod
    def predict_target_time(self, target_phase, **kwargs):
        pass

class BarrierPredictor(PhasePredictor):
    def __init__(self):
        self.timestamp_history = []
        self.phase_history = []

        self.barrier_phase = None
        self.target_phase = None

    def update_phase(self, current_phase, timestamp):
        self.phase_history.append(current_phase)
        self.timestamp_history.append(timestamp)
        if len(self.phase_history) > Config.Gating.PHASE_HISTORY_LENGTH:
            self.phase_history.pop(0)
            self.timestamp_history.pop(0)

    def predict_target_time(self, target_phase, barrier_phase):
        """
        Predicts the future timestamp of the target cardiac phase 
        using linear extrapolation on unwrapped phase data.
        """
        if len(self.phase_history) < Config.Gating.MIN_HISTORY_FOR_PREDICTION:
            return None

        unwrapped_phases = np.unwrap(self.phase_history)
        current_unwrapped = unwrapped_phases[-1]
        current_wrapped = self.phase_history[-1]

        dist_from_barrier = (current_wrapped - barrier_phase) % TWO_PI
        current_barrier_unwrapped = current_unwrapped - dist_from_barrier

        indices = [
            i for i, p in enumerate(unwrapped_phases) 
            if p >= current_barrier_unwrapped
        ]
        
        if len(indices) < Config.Gating.MIN_FRAMES_FOR_PREDICTION:
            return None

        fit_times = np.array([self.timestamp_history[i] for i in indices])
        fit_phases = np.array([unwrapped_phases[i] for i in indices])
        
        slope, intercept = np.polyfit(fit_times, fit_phases, 1)

        if slope <= 1e-6:
            return None

        phase_dist_to_target = (target_phase - current_wrapped) % TWO_PI
        target_unwrapped = current_unwrapped + phase_dist_to_target
        
        predicted_time = (target_unwrapped - current_unwrapped) / slope + self.timestamp_history[-1]
        
        return predicted_time
