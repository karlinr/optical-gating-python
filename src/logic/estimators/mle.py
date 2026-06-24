from loguru import logger
import numpy as np

from logic.estimators.base import register_estimator, PhaseEstimator
from utils.metrics import chi_sq
from utils.fitters import estimate_phase_from_scores
from app.config import Config
from app.data_manager import data_manager
from logic.drift_corrector import DriftCorrector, shift_frame
from scipy.ndimage import gaussian_filter1d

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
        self.log_variance_terms = None
        
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
        
        if Config.Gating.MLE_MODEL_DRIFT_CORRECT:
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
        else:
            raw_frames = np.array([h[0] for h in clean_history], dtype=np.float32)

        block = raw_frames.reshape(n_bins, f_per_bin, *raw_frames.shape[1:])

        self.binned_frames = np.zeros((n_bins, *raw_frames.shape[1:]), dtype=np.float32)
        self.noise_estimate = np.zeros_like(self.binned_frames)

        # Fits a polynomial to each pixel within each bin
        deg = getattr(Config.Gating, "MLE_MODEL_POLY_DEGREE", 1)
        t_centered = np.arange(f_per_bin, dtype=np.float32) - (f_per_bin - 1) / 2.0
    
        for b in range(n_bins):
            block_b = block[b]
            y = block_b.reshape(f_per_bin, -1)
            
            # Fit the polynomial to the pixel time series for the current bin
            coeffs = np.polyfit(t_centered, y, deg)
            
            # Extract the constant term
            self.binned_frames[b] = coeffs[-1].reshape(block_b.shape[1:])
            
            # Compute the fitted values and residuals for variance estimation
            fitted = np.zeros_like(y)
            for power in range(deg + 1):
                fitted += coeffs[power][None, :] * (t_centered[:, None] ** (deg - power))
                
            dof = np.maximum(1, f_per_bin - deg - 1)
            var_b = np.sum((y - fitted) ** 2, axis=0) / dof

            self.noise_estimate[b] = np.maximum(var_b.reshape(block_b.shape[1:]), float(Config.Gating.MLE_MIN_NOISE))
        
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

        # Compute log variance terms for later use in chi-squared calculations
        self.log_variance_terms = np.sum(np.log(self.noise_estimate), axis=(1, 2))

        self._ready = True
        self.frame_history = []

        logger.info("MLE model bootstrapped successfully.")

    def estimate(self, frame):
        corrected_binned = self.drift_corrector.adjust_reference_array(self.binned_frames)
        corrected_noise = self.drift_corrector.adjust_reference_array(self.noise_estimate)
        corrected_frame = self.drift_corrector.adjust_live_frame(frame)

        scores = chi_sq(corrected_frame, corrected_binned, corrected_noise) + self.log_variance_terms
        n_bins = len(scores)
        best_idx = np.argmin(scores)
        self._last_best_idx = best_idx

        fit_res = estimate_phase_from_scores(scores, best_idx, Config.Gating.MLE_FITTER, Config.Gating.MLE_FIT_POINTS, poly_degree=Config.Gating.MLE_POLY_DEGREE, reference_period=n_bins)
        phase_radians = fit_res["phase"]
        vertex_offset = fit_res["vertex_offset"]
        score = fit_res["minimized_score"]
        uncertainty_bins = fit_res["uncertainty"]
        reduced_chi_squared = score / (corrected_frame.size - 1)

        # Scale the uncertainty estimate based on the reduced chi-squared value to account for model fit quality
        if uncertainty_bins is not None:
            uncertainty_radians = uncertainty_bins * (2 * np.pi / n_bins)
            uncertainty_scale = np.sqrt(max(0.0, reduced_chi_squared))
            uncertainty_radians = uncertainty_radians * uncertainty_scale
        else:
            uncertainty_radians = float('inf')

        drift_x, drift_y = self.drift_corrector.drift_x, self.drift_corrector.drift_y
        current_best_match = self.binned_frames[best_idx]
        self.drift_corrector.add_sample(frame, best_match=current_best_match)
        
        return {
            "phase": phase_radians,
            "target_phase": self.target_phase,
            "barrier_phase": self.barrier_phase,
            "metrics": {
                "reduced_chi_squared": reduced_chi_squared - self.log_variance_terms[best_idx] / corrected_frame.size,
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