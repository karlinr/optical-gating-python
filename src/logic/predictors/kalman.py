import numpy as np
from loguru import logger
from app.config import Config

from logic.predictors.base import PhasePredictor, register_predictor
    
@register_predictor("KALMAN")
class KalmanPredictor(PhasePredictor):
    def __init__(self):
        self.X = np.zeros((2, 1))
        self.P = np.array([[1, 0], [0, 1]])
        self.R = np.array([Config.Gating.KALMAN_MEASUREMENT_NOISE])
        self.H = np.array([[1, 0]])

        self.last_timestamp = 0
        self._is_initialized = False

    def _predict(self, dt):
        self.F = np.array([[1, dt], [0, 1]])
        self.Q = np.array([
            [dt**3 / 3.0, dt**2 / 2.0], 
            [dt**2 / 2.0, dt]
        ]) * Config.Gating.KALMAN_PROCESS_NOISE

        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q


    def update_phase(self, current_phase, timestamp, **kwargs):
        if not self._is_initialized:
            self.last_timestamp = timestamp
            self.X[0, 0] = current_phase
            self._is_initialized = True
            logger.info("Kalman predictor initialized.")
            return
        
        self.dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp

        if self.dt <= 0:
            logger.warning(f"Non-positive time delta ({self.dt:.4f}s) detected. Skipping Kalman update.")
            return
        
        uncertainty = kwargs.get("uncertainty_estimate", None)
        if uncertainty is not None:
            self.R = np.array([uncertainty**2])
        else:
            self.R = np.array([Config.Gating.KALMAN_MEASUREMENT_NOISE])

        self._predict(self.dt)

        predicted_wrapped_phase = self.X[0, 0] % (2 * np.pi)
        phase_residual = (current_phase - predicted_wrapped_phase + np.pi) % (2 * np.pi) - np.pi
        residual = np.array([[phase_residual]])

        self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.X = self.X + self.K @ residual
        self.P = (np.eye(2) - self.K @ self.H) @ self.P

    def predict_target_time(self, target_phase, **kwargs):
        # We need to predict the time until the target phase is reached, given the current state estimate
        if self.X[1, 0] <= 1e-6 and "reference_period" in kwargs:
            ref_period = kwargs["reference_period"]
            framerate = getattr(Config.Cameras.BF, "framerate", 80)
            if ref_period > 0 and framerate > 0:
                est_period_s = ref_period / framerate
                self.X[1, 0] = (2 * np.pi) / est_period_s
                logger.info(f"Kalman phase velocity warm-started: {self.X[1, 0]:.4f} rad/s (Period: {est_period_s:.4f}s)")

        current_phase_estimate = self.X[0, 0] % (2 * np.pi)
        phase_diff = (target_phase - current_phase_estimate) % (2 * np.pi)
        
        if self.X[1, 0] <= 1e-6:
            return None
            
        time_to_target = phase_diff / self.X[1, 0]
        est_heart_period_s = 2 * np.pi / self.X[1, 0]
        
        return {
            "predicted_time_rel": time_to_target,
            "metrics": {
                "est_period": est_heart_period_s,
                "phase_estimate": current_phase_estimate,
                "phase_velocity_estimate": self.X[1, 0]
            }
        }