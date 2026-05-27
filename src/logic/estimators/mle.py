from loguru import logger
import numpy as np

# Ensure we use the optimized utils
from logic.estimators.base import register_estimator, PhaseEstimator
from logic.utils import chi_sq 
from app.config import Config
from app.data_manager import data_manager

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
        n_bins = Config.Gating.MLE_BINS
        frames = np.stack([h[0] for h in self.frame_history])
        phases = np.array([h[1] for h in self.frame_history])

        bins = np.linspace(0, 2 * np.pi, n_bins + 1)
        bin_indices = np.digitize(phases, bins) - 1
        bin_indices[bin_indices == n_bins] = 0

        logger.info(f"Frame count per bin: \n{np.bincount(bin_indices, minlength=n_bins)}")

        frame_shape = frames[0].shape
        self.binned_frames = np.zeros((n_bins, *frame_shape), dtype=np.float32)
        self.noise_estimate = np.zeros((n_bins, *frame_shape), dtype=np.float32)

        for b in range(n_bins):
            mask = (bin_indices == b)

            masked_frames = frames[mask]
            masked_phases = phases[mask]

            self.binned_frames[b] = np.mean(masked_frames, axis=0) if np.any(mask) else np.zeros(frame_shape, dtype=np.float32)

            if len(masked_frames) > 1:
                order = np.argsort(masked_phases)
                sorted_frames = masked_frames[order]
                diffs = np.diff(sorted_frames, axis=0)
                self.noise_estimate[b] = np.sum(diffs ** 2, axis=0) / (2 * (len(masked_frames) - 1))
            else:
                logger.warning(f"Bin {b} has only one frame. Setting noise estimate to default value.")
                self.noise_estimate[b] = np.ones(frame_shape, dtype=np.float32) * Config.Gating.MLE_MIN_NOISE

            # Set any zero noise estimates to a minimum value to avoid division issues
            self.noise_estimate[b][self.noise_estimate[b] < Config.Gating.MLE_MIN_NOISE] = Config.Gating.MLE_MIN_NOISE

        logger.info("MLE Estimator model built. Ready for estimation.")

        data_manager.save("binned_frames", self.binned_frames.copy())
        data_manager.save("noise_estimate", self.noise_estimate.copy())
     
        self._ready = True
        self.frame_history = []

    def estimate(self, frame):
        # Calculate chi-squared scores
        scores = chi_sq(frame, self.binned_frames, self.noise_estimate)
        n_bins = len(scores)
        best_idx = np.argmin(scores)
        reduced_chi_squared = scores[best_idx] / frame.size

        fit_points = Config.Gating.MLE_FIT_POINTS
        
        x = np.arange(-fit_points, fit_points + 1)
        indices = np.arange(best_idx - fit_points, best_idx + fit_points + 1)
        y_fit = np.take(scores, indices, mode='wrap')

        try:
            coeffs = np.polyfit(x, y_fit, 2)
            a, b = coeffs[0], coeffs[1]
            
            vertex_offset = -b / (2 * a)
            # Clamp the offset to the fitting window to prevent exploding phase values
            vertex_offset = np.clip(vertex_offset, -fit_points, fit_points)
        except (np.linalg.LinAlgError, TypeError):
            vertex_offset = 0.0
            a = 1.0
            logger.warning("MLE parabola fitting failed. Defaulting vertex offset to 0.")

        logger.debug(f"MLE Estimate: Best Bin={best_idx}, Offset={vertex_offset:.2f}, Reduced Chi-Squared={reduced_chi_squared:.2f}")

        phase_radians = ((best_idx + vertex_offset + 0.5) % n_bins / n_bins) * 2 * np.pi
        
        # Uncertainty is the curvature of the fit
        # For a parabola the curvature is given by 2 * a
        uncertainty_radians = np.sqrt(2 / a) * (2 * np.pi / n_bins)
        
        return {
            "phase": phase_radians,
            "target_phase": self.target_phase,
            "barrier_phase": self.barrier_phase,
            "metrics": {
                "reduced_chi_squared": reduced_chi_squared,
                "uncertainty_estimate": uncertainty_radians,
                "best_index": best_idx,
                "reference_period": n_bins,
                "vertex_offset": vertex_offset
            }
        }
    
    @property
    def active_dependencies(self):
        # Once MLE is operational, drop the dependency to prevent heavy SAD processing loops
        return [] if self.is_ready() else self.dependencies