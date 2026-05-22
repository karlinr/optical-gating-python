from collections import deque

from loguru import logger
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
            
    def predict_target_time(self, target_phase, barrier_phase, best_index, reference_period):
        """
        Predicts the future timestamp of the target cardiac phase 
        using linear extrapolation on unwrapped phase data, using a clamped
        fitting history window to eliminate the prediction blind spot.
        """
        if len(self.phase_history) < Config.Gating.MIN_HISTORY_FOR_PREDICTION:
            return None

        unwrapped_phases = np.unwrap(self.phase_history)
        current_unwrapped = unwrapped_phases[-1]
        current_wrapped = self.phase_history[-1]

        barrier_index = int(np.round((barrier_phase / (2 * np.pi)) * reference_period))
        
        frames_since_barrier = int((best_index - barrier_index) % reference_period)
        
        n_points = frames_since_barrier
        min_frames = Config.Gating.MIN_FRAMES_FOR_PREDICTION
        max_frames = getattr(Config.Gating, "MAX_FRAMES_FOR_PREDICTION", 25)
        
        n_points = max(n_points, min_frames)
        n_points = min(n_points, max_frames, len(self.phase_history))

        if not self._validate_timeline_continuity(n_points):
            return None

        fit_times = np.array(self.timestamp_history[-n_points:])
        fit_phases = unwrapped_phases[-n_points:]
        
        slope, intercept = np.polyfit(fit_times, fit_phases, 1)

        if slope <= 1e-6:
            return None

        fitted_current_wrapped = (slope * self.timestamp_history[-1] + intercept) % (2 * np.pi)
        phase_dist_to_target = (target_phase - fitted_current_wrapped) % (2 * np.pi)
        predicted_time = phase_dist_to_target / slope
        est_heart_period_s = 2 * np.pi / slope
        
        return {
            "predicted_time_rel": predicted_time,
            "metrics": {
                "est_period": est_heart_period_s,
                "n_points": n_points,
                "slope": slope,
                "intercept": intercept
            }
        }

    def _validate_timeline_continuity(self, n_points: int):
        """Internal helper to ensure no frame drops occurred in the fitting window."""
        if n_points <= 1:
            return True
            
        active_timestamps = np.array(self.timestamp_history[-n_points:])
        tsdiffs = active_timestamps[1:] - active_timestamps[:-1]
        
        if np.max(tsdiffs) > np.min(tsdiffs) * 2.5:
            logger.debug(f"Linear regression aborted: Frame drop detected in history window.")
            return False
        return True
    
class TriggerDecider:
    def __init__(self):
        self.frame_interval = 1.0 / Config.Cameras.BF.framerate
        self.most_recent_trigger_time = -10000
        self.timestamp_history = deque(maxlen = 100)
    
    def evaluate_trigger(self, current_time, predicted_time, est_period):
        """
        Translates absolute predicted targets into relative lookahead values 
        to evaluate the original spim-interface triggering criteria.
        
        Returns:
            (bool, float): A tuple containing a flag indicating whether to fire, 
                           and the final adjusted relative time to wait.
        """
        time_to_wait_s = predicted_time - current_time

        if self.most_recent_trigger_time >= current_time - (est_period / 2.0):
            logger.debug("Trigger rejected: Already issued on this cardiac cycle.")
            # Coarsely shift target prediction to the next expected heartbeat cycle
            time_to_wait_s += est_period
            return False, time_to_wait_s

        if time_to_wait_s < Config.Gating.PREDICTION_LATENCY:
            # Panic mode: Target is extremely close, but fire immediately anyway
            logger.warning(f"Panic trigger issued! Lookahead ({time_to_wait_s:.4f}s) is below latency floor.")
            self.most_recent_trigger_time = current_time + time_to_wait_s
            return True, time_to_wait_s
            
        if (time_to_wait_s - (Config.Gating.EXTRAPOLATION_FACTOR * self.frame_interval)) < Config.Gating.PREDICTION_LATENCY:
            # Standard commit window: Not enough time to wait for the next frame's data
            logger.debug(f"Standard trigger committed. Lookahead: {time_to_wait_s:.4f}s.")
            self.most_recent_trigger_time = current_time + time_to_wait_s
            return True, time_to_wait_s

        logger.debug(f"Hold trigger: Lookahead ({time_to_wait_s:.4f}s) allows waiting for the next frame.")
        return False, time_to_wait_s

    def handle_hardware_rejection(self, current_time, est_period):
        """
        Forces a software cooldown following a hardware timing box rejection.
        Prevents rapid-fire panic cycles within a heartbeat that's already passed.
        """
        logger.warning(
            f"Hardware collision detected! Forcing a software lockout cooldown "
            f"for the remaining cycle duration ({est_period:.4f}s)."
        )
        # Advance the trigger lockout to the future to block subsequent frames in this cycle
        self.most_recent_trigger_time = current_time + (est_period * 0.5)