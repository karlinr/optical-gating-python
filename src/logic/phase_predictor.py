import numpy as np
from app.config import Config
from abc import ABC, abstractmethod

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

        indices = [i for i, p in enumerate(self.phase_history) if p >= barrier_phase]
        
        if len(indices) < Config.Gating.MIN_FRAMES_FOR_PREDICTION:
            return None

        fit_times = np.array([self.timestamp_history[i] for i in indices])
        fit_phases = np.array([unwrapped_phases[i] for i in indices])
        
        slope, intercept = np.polyfit(fit_times, fit_phases, 1)

        # Ensure the heart is actually moving forward
        if slope <= 1e-6:
            return None

        two_pi = 2 * np.pi
        phase_dist_to_target = (target_phase - current_wrapped) % two_pi
        
        target_unwrapped = current_unwrapped + phase_dist_to_target
        
        predicted_time = (target_unwrapped - intercept) / slope
        
        return predicted_time
