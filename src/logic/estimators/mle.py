from loguru import logger
import numpy as np
import time

from logic.estimators.base import register_estimator, PhaseEstimator
from logic.utils import chi_sq 
from app.config import Config
from app.data_manager import data_manager
from logic.drift_corrector import DriftCorrector  # Import the new utility

@register_estimator("MLE")
class MLEEstimator(PhaseEstimator):
    dependencies = ["SAD"]

    def __init__(self):
        super().__init__()
        self.binned_frames = None
        self.noise_estimate = None
        self.frame_history = []
        self.target_phase = None
        self.barrier_phase = None
        
        self.drift_corrector = DriftCorrector()
        self._last_best_idx = None

    def update(self, frame, **kwargs):
        if not self.is_ready():
            context = kwargs.get("context", {})
            sad_result = context.get("SAD")
            phase = sad_result.get("phase") if sad_result else None
            if phase is not None:
                self.frame_history.append((frame, phase))

                if sad_result.get("target_phase") is not None:
                    self.target_phase = sad_result["target_phase"]
                if sad_result.get("barrier_phase") is not None:
                    self.barrier_phase = sad_result["barrier_phase"]

                if len(self.frame_history) >= Config.Gating.MLE_BOOTSTRAP_FRAMES:
                    logger.info("Sufficient samples for MLE. Building model.")
                    self.build_model()

        if self.is_ready():
            return self.estimate(frame)
        
        return None

    def build_model(self):
        start_time = time.time()
        logger.info("Bootstrapping MLE model with collected frames...")
        n_bins = Config.Gating.MLE_BINS
        f_per_bin = len(self.frame_history) // n_bins
        self.frame_history.sort(key=lambda x: x[1])
        clean_history = self.frame_history[:n_bins * f_per_bin]
        raw_frames = np.array([h[0] for h in clean_history], dtype=np.float32)
        block = raw_frames.reshape(n_bins, f_per_bin, *raw_frames.shape[1:])
        self.binned_frames = np.mean(block, axis=1)
        sum_sq_diff = np.sum(np.diff(block, axis=1) ** 2, axis=1)
        self.noise_estimate = np.maximum(sum_sq_diff / (2 * (f_per_bin - 1)), Config.Gating.MLE_MIN_NOISE)
        data_manager.save("binned_frames", self.binned_frames.copy())
        data_manager.save("noise_estimate", self.noise_estimate.copy())
        self._ready = True
        self.frame_history = []
        logger.info(f"MLE model built in {time.time() - start_time:.2f} seconds with {len(clean_history)} frames.")

    def estimate(self, frame):
        if self._last_best_idx is not None:
            prev_best_match = self.binned_frames[self._last_best_idx]
            self.drift_corrector.add_sample(frame, best_match=prev_best_match)

        corrected_binned = self.drift_corrector.adjust_reference_array(self.binned_frames)
        corrected_noise = self.drift_corrector.adjust_reference_array(self.noise_estimate)
        corrected_frame = self.drift_corrector.adjust_live_frame(frame)

        scores = chi_sq(corrected_frame, corrected_binned, corrected_noise)
        n_bins = len(scores)
        best_idx = np.argmin(scores)
        self._last_best_idx = best_idx  # Keep tracking reference index up to date

        fit_points = Config.Gating.MLE_FIT_POINTS
        
        x = np.arange(-fit_points, fit_points + 1)
        indices = np.arange(best_idx - fit_points, best_idx + fit_points + 1)
        y_fit = np.take(scores, indices, mode='wrap')

        try:
            a, b, c = np.polyfit(x, y_fit, 2)
            
            vertex_offset = -b / (2 * a)
            vertex_offset = np.clip(vertex_offset, -fit_points, fit_points)

            minimized_score = a * (vertex_offset ** 2) + b * vertex_offset + c

            if minimized_score < 0:
                logger.warning(f"Minima is negative ({minimized_score:.2f}), setting to {scores[best_idx]:.2f}.")
                minimized_score = scores[best_idx]

            reduced_chi_squared = minimized_score / corrected_frame.size
        except (np.linalg.LinAlgError, TypeError):
            vertex_offset = 0.0
            a = 1.0
            reduced_chi_squared = scores[best_idx] / corrected_frame.size
            logger.warning("MLE parabola fitting failed. Defaulting vertex offset to 0.")

        logger.debug(
            f"MLE Estimate: Best Bin={best_idx}, Offset={vertex_offset:.2f}, "
            f"Reduced Chi2={reduced_chi_squared:.2f} | Drift=({self.drift_corrector.drift[0]},{self.drift_corrector.drift[1]})"
        )

        phase_radians = ((best_idx + vertex_offset + 0.5) % n_bins / n_bins) * 2 * np.pi
        uncertainty_radians = np.sqrt(1 / a) * (2 * np.pi / n_bins)
        
        return {
            "phase": phase_radians,
            "target_phase": self.target_phase,
            "barrier_phase": self.barrier_phase,
            "metrics": {
                "reduced_chi_squared": reduced_chi_squared,
                "uncertainty_estimate": uncertainty_radians,
                "best_index": best_idx,
                "reference_period": n_bins,
                "vertex_offset": vertex_offset,
                "scores": scores,
                "drift_x": self.drift_corrector.drift[0],
                "drift_y": self.drift_corrector.drift[1]
            }
        }
    
    @property
    def active_dependencies(self):
        return [] if self.is_ready() else self.dependencies