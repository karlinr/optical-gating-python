import numpy as np
from loguru import logger
import time

from logic.estimators.mle import MLEEstimator
from logic.estimators.base import register_estimator
from app.config import Config
from app.data_manager import data_manager

@register_estimator("MLE_ANOMALY")
class MLEAnomalyEstimator(MLEEstimator):
    def __init__(self):
        super().__init__()
        self.residual_mean = None
        self.residual_std = None
        self.residual_median = None
        self.residual_mad = None

        self.mu_z_macro_mean = None
        self.mu_z_macro_std = None
        self.median_z_macro_mean = None
        self.median_z_macro_std = None

    def build_model(self):
        if len(self.frame_history) < Config.Gating.MLE_BOOTSTRAP_FRAMES:
            logger.error("Not enough collected frames to build MLE_ANOMALY model.")
            return

        logger.info("Bootstrapping MLE_ANOMALY model with collected frames...")
        start_time = time.time()

        # Frustratingly we have to do the binning and reshaping again
        # In future we'll refactor this but for now we'll stick for this for testing
        n_bins = Config.Gating.MLE_BINS
        f_per_bin = len(self.frame_history) // n_bins

        self.frame_history.sort(key=lambda x: x[1])
        clean_history = self.frame_history[:n_bins * f_per_bin]
        raw_frames = np.array([h[0] for h in clean_history], dtype=np.float32)
        block = raw_frames.reshape(n_bins, f_per_bin, *raw_frames.shape[1:])

        super().build_model()

        # Calculate residuals
        #exp_expanded = self.binned_frames[:, np.newaxis, :, :]
        #noise_expanded = self.noise_estimate[:, np.newaxis, :, :]
        self.bin_std_dev = np.maximum(np.std(block, axis=1), 1)
        exp_expanded = self.binned_frames[:, np.newaxis, :, :]
        std_expanded = self.bin_std_dev[:, np.newaxis, :, :]
        #residuals = (block - exp_expanded) / np.sqrt(noise_expanded)
        residuals = (block - exp_expanded) / std_expanded

        # Compute statistics across the time dimension for each bin
        self.residual_mean = np.mean(residuals, axis=1)
        self.residual_std = np.maximum(np.std(residuals, axis=1), 1e-2)
        self.residual_median = np.median(residuals, axis=1)
        abs_deviation = np.abs(residuals - self.residual_median[:, np.newaxis, :, :])
        self.residual_mad = np.maximum(np.median(abs_deviation, axis=1), 1e-2)

        logger.info("Building Layer 3 temporal calibration baseline for location metrics...")
        self.mu_z_macro_mean = np.zeros(n_bins)
        self.mu_z_macro_std = np.zeros(n_bins)
        self.median_z_macro_mean = np.zeros(n_bins)
        self.median_z_macro_std = np.zeros(n_bins)

        for b in range(n_bins):
            bin_residuals = residuals[b]
            
            b_mean = self.residual_mean[b][np.newaxis, :, :]
            b_std = self.residual_std[b][np.newaxis, :, :]
            b_median = self.residual_median[b][np.newaxis, :, :]
            b_mad = self.residual_mad[b][np.newaxis, :, :]
            
            bin_z_map_std = (bin_residuals - b_mean) / b_std
            bin_z_map_robust = (bin_residuals - b_median) / (b_mad * 1.4826)
            
            f_mu_zs = np.mean(bin_z_map_std, axis=(1, 2))
            f_median_zs = np.median(bin_z_map_robust, axis=(1, 2))
            
            self.mu_z_macro_mean[b] = np.mean(f_mu_zs)
            self.mu_z_macro_std[b] = np.maximum(np.std(f_mu_zs), 0.1)
            
            self.median_z_macro_mean[b] = np.mean(f_median_zs)
            self.median_z_macro_std[b] = np.maximum(np.std(f_median_zs), 0.1)

        # Save
        data_manager.save("residual_mean", self.residual_mean.copy())
        data_manager.save("residual_std", self.residual_std.copy())
        data_manager.save("residual_median", self.residual_median.copy())
        data_manager.save("residual_mad", self.residual_mad.copy())

        logger.info(f"MLE_ANOMALY statistical baseline built in {time.time() - start_time:.2f} seconds.")

    def estimate(self, frame):
        res = super().estimate(frame)
        if res is None:
            return None
        
        # Extract the best matching bin and the drift used for this estimation
        best_idx = res["metrics"]["best_index"]
        drift_used = (res["metrics"]["drift_x"], res["metrics"]["drift_y"])

        # Expected values and noise estimates for the best matching bin
        exp = self.drift_corrector.adjust_reference_array(self.binned_frames[best_idx], drift=drift_used)
        var = self.drift_corrector.adjust_reference_array(self.noise_estimate[best_idx], drift=drift_used)
        true_std = self.drift_corrector.adjust_reference_array(self.bin_std_dev[best_idx], drift=drift_used)
        mean = self.drift_corrector.adjust_reference_array(self.residual_mean[best_idx], drift=drift_used)
        std = self.drift_corrector.adjust_reference_array(self.residual_std[best_idx], drift=drift_used)
        median = self.drift_corrector.adjust_reference_array(self.residual_median[best_idx], drift=drift_used)
        mad = self.drift_corrector.adjust_reference_array(self.residual_mad[best_idx], drift=drift_used)

        # Actual values for the current frame
        obs = self.drift_corrector.adjust_live_frame(frame, drift=drift_used)
        #raw_residual = (obs - exp) / np.sqrt(var)
        raw_residual = (obs - exp) / true_std

        # And the z-scores for both standard and robust metrics
        z_map_std = (raw_residual - mean) / std
        z_map_robust = (raw_residual - median) / (mad * 1.4826)
        mu_z = (np.mean(z_map_std) - self.mu_z_macro_mean[best_idx]) / self.mu_z_macro_std[best_idx]
        sigma_z = np.std(z_map_std)
        median_z = (np.median(z_map_robust) - self.median_z_macro_mean[best_idx]) / self.median_z_macro_std[best_idx]
        mad_z = np.median(np.abs(z_map_robust - median_z)) * 1.4826

        res["metrics"].update({
            "mu_z": mu_z,
            "sigma_z": sigma_z,
            "median_z": median_z,
            "mad_z": mad_z,
            "diff_mean_median": mu_z - median_z,
            "diff_std_mad": sigma_z - mad_z
        })

        return res