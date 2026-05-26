import numpy as np
from loguru import logger
from app.config import Config

from logic.predictors.base import PhasePredictor, register_predictor
    

@register_predictor("BARRIER")
class BarrierPredictor(PhasePredictor):
    def __init__(self):
        self.timestamp_history = []
        self.phase_history = []

    def update_phase(self, current_phase, timestamp, **kwargs):
        self.phase_history.append(current_phase)
        self.timestamp_history.append(timestamp)
        if len(self.phase_history) > Config.Gating.PHASE_HISTORY_LENGTH:
            self.phase_history.pop(0)
            self.timestamp_history.pop(0)
            
    def predict_target_time(self, target_phase, **kwargs):
        """
        Predicts the future timestamp of the target cardiac phase 
        using linear extrapolation on unwrapped phase data, using a clamped
        fitting history window to eliminate the prediction blind spot.
        """
        if len(self.phase_history) < Config.Gating.MIN_HISTORY_FOR_PREDICTION:
            return None

        unwrapped_phases = np.unwrap(self.phase_history)

        barrier_phase = kwargs.get("barrier_phase")
        best_index = kwargs.get("best_index")
        reference_period = kwargs.get("reference_period")

        barrier_index = int(np.round((barrier_phase / (2 * np.pi)) * reference_period))
        
        frames_since_barrier = int((best_index - barrier_index) % reference_period)
        
        n_points = frames_since_barrier
        min_frames = Config.Gating.MIN_FRAMES_FOR_PREDICTION
        max_frames = Config.Gating.MAX_FRAMES_FOR_PREDICTION
        
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