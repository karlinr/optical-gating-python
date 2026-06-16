from loguru import logger
import numpy as np

from logic.estimators.base import register_estimator, PhaseEstimator
from logic.utils import chi_sq 
from app.config import Config
from app.data_manager import data_manager
from logic.drift_corrector import DriftCorrector, shift_frame
from scipy.ndimage import gaussian_filter1d
from numba import njit, prange

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
                metrics = sad_result.get("metrics", {})
                drift_x = metrics.get("drift_x", 0.0)
                drift_y = metrics.get("drift_y", 0.0)
                self.frame_history.append((frame, phase, drift_x, drift_y))

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
        logger.info("Bootstrapping MLE model with collected frames...")

        # Record the drift of the last bootstrap sample before sorting by phase
        if self.frame_history:
            last_frame_drift_x = self.frame_history[-1][2]
            last_frame_drift_y = self.frame_history[-1][3]
        else:
            last_frame_drift_x = 0.0
            last_frame_drift_y = 0.0

        # Apply smoothing to the frame history phase estimates to reduce noise before binning
        if Config.Gating.MLE_PHASE_SMOOTHING_SIGMA > 0:
            phases = np.array([h[1] for h in self.frame_history])
            cos_p = gaussian_filter1d(np.cos(phases), sigma=Config.Gating.MLE_PHASE_SMOOTHING_SIGMA, mode='nearest')
            sin_p = gaussian_filter1d(np.sin(phases), sigma=Config.Gating.MLE_PHASE_SMOOTHING_SIGMA, mode='nearest')
            smoothed_phases = np.arctan2(sin_p, cos_p) % (2 * np.pi)
            self.frame_history = [(h[0], sp, h[2], h[3]) for h, sp in zip(self.frame_history, smoothed_phases)]          

        n_bins = Config.Gating.MLE_BINS
        f_per_bin = len(self.frame_history) // n_bins
        self.frame_history.sort(key=lambda x: x[1])

        clean_history = self.frame_history[:n_bins * f_per_bin]
        
        # Shift frames based on drift
        shifted_frames = []
        for h in clean_history:
            frame, phase, dx, dy = h
            if Config.Gating.DRIFT_CORRECT and Config.Gating.MLE_MODEL_DRIFT_CORRECT:
                shifted_f = shift_frame(frame, dx, dy)
            else:
                shifted_f = frame
            shifted_frames.append(shifted_f)

        raw_frames = np.array(shifted_frames, dtype=np.float32)

        block = raw_frames.reshape(n_bins, f_per_bin, *raw_frames.shape[1:])

        # Get model
        self.binned_frames = np.mean(block, axis=1)

        # Get noise model
        sum_sq_diff = np.sum(np.diff(block, axis=1) ** 2, axis=1)
        self.noise_estimate = np.maximum(sum_sq_diff / (2 * (f_per_bin - 1)), float(Config.Gating.MLE_MIN_NOISE))
        
        if Config.Gating.MLE_SMOOTHING_SIGMA > 0:
            self.binned_frames = gaussian_filter1d(self.binned_frames, sigma=Config.Gating.MLE_SMOOTHING_SIGMA, axis=0, mode='wrap')
            self.noise_estimate = gaussian_filter1d(self.noise_estimate, sigma=Config.Gating.MLE_SMOOTHING_SIGMA, axis=0, mode='wrap')

        data_manager.save("binned_frames", self.binned_frames.copy())
        data_manager.save("noise_estimate", self.noise_estimate.copy())

        # Sync the drift corrector state to establish correct cropping windows for estimation
        if Config.Gating.DRIFT_CORRECT:
            self.drift_corrector.drift_x = last_frame_drift_x
            self.drift_corrector.drift_y = last_frame_drift_y
            self.drift_corrector.mx = int(np.ceil(abs(last_frame_drift_x))) + 1
            self.drift_corrector.my = int(np.ceil(abs(last_frame_drift_y))) + 1

        self._ready = True
        self.frame_history = []

        logger.info("MLE model bootstrapped successfully.")

    def estimate(self, frame):
        corrected_binned = self.drift_corrector.adjust_reference_array(self.binned_frames)
        corrected_noise = self.drift_corrector.adjust_reference_array(self.noise_estimate)
        corrected_frame = self.drift_corrector.adjust_live_frame(frame)

        scores = chi_sq(corrected_frame, corrected_binned, corrected_noise)
        n_bins = len(scores)
        best_idx = np.argmin(scores)
        self._last_best_idx = best_idx

        fit_points = Config.Gating.MLE_FIT_POINTS
        
        x = np.arange(-fit_points, fit_points + 1)
        indices = np.arange(best_idx - fit_points, best_idx + fit_points + 1)
        y_fit = np.take(scores, indices, mode='wrap')

        try:
            a, b, c = np.polyfit(x, y_fit, 2)
            
            vertex_offset = -b / (2 * a)

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
            f"Reduced Chi2={reduced_chi_squared:.2f} | Drift=({self.drift_corrector.drift_x},{self.drift_corrector.drift_y})"
        )

        phase_radians = ((best_idx + vertex_offset) % n_bins / n_bins) * 2 * np.pi
        if a > 0:
            uncertainty_radians = np.sqrt(1 / a) * (2 * np.pi / n_bins)
        else:
            uncertainty_radians = np.pi
            logger.warning("MLE parabola is not convex. Uncertainty set to infinity.")

        drift_x, drift_y = self.drift_corrector.drift_x, self.drift_corrector.drift_y
        current_best_match = self.binned_frames[best_idx]
        self.drift_corrector.add_sample(frame, best_match=current_best_match)
        
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
                "drift_x": drift_x,
                "drift_y": drift_y
            }
        }
    
    @property
    def active_dependencies(self):
        return [] if self.is_ready() else self.dependencies