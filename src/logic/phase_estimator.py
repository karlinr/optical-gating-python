from loguru import logger
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List, Any

# Ensure we use the optimized utils
from logic.utils import v_fitting, chi_sq, sad_with_references 
from app.config import Config

TWO_PI = 2 * np.pi
NUM_EXTRA_REF_FRAMES = 2

class PhaseEstimator(ABC):
    def __init__(self):
        self._ready = False

    def is_ready(self):
        return self._ready

    @abstractmethod
    def add_sample(self, frame, **kwargs):
        pass

    @abstractmethod
    def build_model(self):
        pass

class SADEstimator(PhaseEstimator):
    """
    Standard SAD-based estimator as used in original optical-gating code
    Much of this is adapted from open-optical-gating @ https://github.com/Glasgow-ICG/open-optical-gating/tree/main
    """

    def __init__(self):
        super().__init__()
        self.reference_frames = None
        self.reference_period = None
        self.barrier_phase = None
        self.frame_history = []
        self.period_history = []

    def add_sample(self, frame, **kwargs):
        """Adds a frame and manages the history buffer based on heart rate."""
        timestamp = kwargs.get("timestamp")
        if timestamp is None:
            logger.error("SADEstimator requires a timestamp in add_sample.")
            return "SAD_ERROR"

        self.frame_history.append((frame, timestamp))

        ref_buffer_duration = self.frame_history[-1][1] - self.frame_history[0][1]
        max_duration = 1.0 / Config.Gating.MIN_HEART_RATE_HZ
        
        while ref_buffer_duration > max_duration:
            self.frame_history.pop(0)
            ref_buffer_duration = self.frame_history[-1][1] - self.frame_history[0][1]

        if len(self.frame_history) > Config.Gating.MIN_PERIOD:
            if self.build_model():
                return "SAD_MODEL_READY"
            
        return "SAD_COLLECTING_FRAMES"

    def build_model(self):
        """Logic to establish period, extract sequence, and set barrier phase."""
        start, stop, period = self._establish_indices()
        
        if start is None:
            return False

        raw_sequence = [f[0] for f in self.frame_history[start:stop]]
        self.reference_frames = np.array(raw_sequence)
        self.reference_period = period

        target_frame, barrier_frame = self._pick_frames()
        
        self.barrier_phase = TWO_PI * (target_frame / self.reference_period)
        self._ready = True
        
        logger.info(f"SAD Model Built: Period={period:.2f}, Barrier Phase={self.barrier_phase:.2f}")
        return True

    def _establish_indices(self):
        """Establish list indices representing a reference period."""
        frame = self.frame_history[-1][0]
        past_frames = np.array([f[0] for f in self.frame_history[:-1]])

        # Calculate Diffs
        diffs = sad_with_references(frame, past_frames)

        # Calculate Period length
        period = self._calculate_period_length(diffs)
        if period != -1:
            logger.info(f"Estimated period: {period:.2f} frames")
            self.period_history.append(period)

        # Stability check: Requires 5 + 2*padding frames of history
        history_stable = (len(self.period_history) >= (5 + (2 * NUM_EXTRA_REF_FRAMES))
                          and period > 6)

        if period != -1 and history_stable:
            period_to_use = self.period_history[-1 - NUM_EXTRA_REF_FRAMES]
            num_refs = int(period_to_use + 1) + (2 * NUM_EXTRA_REF_FRAMES)
            
            start = len(past_frames) - num_refs
            stop = len(past_frames)
            return start, stop, period_to_use

        return None, None, None

    def _calculate_period_length(self, diffs):
        """Interpolated period search based on threshold factors."""
        if diffs.size < Config.Gating.MIN_PERIOD:
            return -1

        min_score = max_score = diffs[-1]
        delta_for_min_since_max = 0
        min_since_max = diffs[-1]
        stage = 1
        got = False

        for d in range(Config.Gating.MIN_PERIOD, diffs.size + 1):
            score = diffs[-d]
            
            lower_thresh = min_score + (max_score - min_score) * Config.Gating.LOWER_THRESHOLD_FACTOR
            upper_thresh = min_score + (max_score - min_score) * Config.Gating.UPPER_THRESHOLD_FACTOR

            if score < lower_thresh and stage == 1:
                stage = 2
            if score > upper_thresh and stage == 2:
                stage = 3
                got = True
                break

            if score > max_score:
                max_score = score
                min_since_max = score
                delta_for_min_since_max = d
                stage = 1
            elif score < min_score:
                min_score = score

            if score < min_since_max:
                min_since_max = score
                delta_for_min_since_max = d

        if got:
            best_match_idx = diffs.size - delta_for_min_since_max
            # Sub-frame correction using v-fitting
            offset, _ = v_fitting(diffs[best_match_idx-1], diffs[best_match_idx], diffs[best_match_idx+1])
            return delta_for_min_since_max - offset
            
        return -1

    def _pick_frames(self):
        """Automatically identify target and barrier frames."""
        # Calculate deltas between consecutive frames in the sequence
        inner_range = len(self.reference_frames) - 2 * NUM_EXTRA_REF_FRAMES
        deltas = np.zeros(inner_range)
        
        for i in range(inner_range):
            f1 = self.reference_frames[i + NUM_EXTRA_REF_FRAMES]
            f2 = self.reference_frames[i + NUM_EXTRA_REF_FRAMES + 1]
            deltas[i] = np.sum(np.abs(f1.astype(np.int32) - f2.astype(np.int32)))

        max_pos = np.argmax(deltas)
        
        # Max change sub-frame estimate
        offset, _ = v_fitting(-deltas[max_pos-1], -deltas[max_pos], -deltas[max_pos+1])
        target_frame = (max_pos + offset + (self.reference_period / 3.0)) % self.reference_period
        
        # Barrier: first point rising past midpoint between min and max deltas
        min_delta, max_delta = np.min(deltas), np.max(deltas)
        midpoint = (min_delta + max_delta) / 2
        barrier_frame = np.argmin(deltas)
        
        while deltas[barrier_frame] < midpoint:
            barrier_frame = (barrier_frame + 1) % int(self.reference_period)
            
        return target_frame, barrier_frame

    def estimate(self, frame):
        """Estimates the phase and returns (phase, score)."""
        scores = sad_with_references(frame, self.reference_frames)
        best_idx = np.argmin(scores[NUM_EXTRA_REF_FRAMES : -NUM_EXTRA_REF_FRAMES]) + NUM_EXTRA_REF_FRAMES
        
        offset, score = v_fitting(scores[best_idx - 1], scores[best_idx], scores[best_idx + 1])
        phase = ((best_idx + offset - NUM_EXTRA_REF_FRAMES) / self.reference_period) * TWO_PI
        
        return phase % TWO_PI, score
    
class MLEEstimator(PhaseEstimator):
    def __init__(self):
        super().__init__()
        self.binned_frames = None
        self.noise_estimate = None
        self.frame_history = []

    def add_sample(self, frame, **kwargs):
        phase = kwargs.get('phase')
        if phase is not None:
            self.frame_history.append((frame, phase))

        if len(self.frame_history) >= Config.Gating.MLE_BOOTSTRAP_FRAMES and not self._ready:
            logger.info("MLE Estimator has enough samples. Building model.")
            self.build_model()
            return "MLE_MODEL_BUILT"
        else:
            return "MLE_COLLECTING_FRAMES"

    def build_model(self):
        # Again, needs to be copied over from my main code.
        # This will bin frames by phase and estimating pixel-wise mean and variance for each bin.
        # The binned frames will be of shape (n_bins, height, width) and the noise estimate will be of shape (n_bins, height, width)
        n_bins = Config.Gating.MLE_BINS
        self.binned_frames = np.zeros((n_bins, *self.frame_history[0][0].shape), dtype=np.float32)
        count_per_bin = np.zeros(n_bins, dtype=np.int32)
        for frame, phase in self.frame_history:
            bin_idx = int((phase / TWO_PI) * n_bins) % n_bins
            self.binned_frames[bin_idx] += frame
            count_per_bin[bin_idx] += 1
        for i in range(n_bins):
            if count_per_bin[i] > 0:
                self.binned_frames[i] /= count_per_bin[i]
        # Estimate noise as the variance within each bin
        self.noise_estimate = np.zeros_like(self.binned_frames)
        for frame, phase in self.frame_history:
            bin_idx = int((phase / TWO_PI) * n_bins) % n_bins
            self.noise_estimate[bin_idx] += (frame - self.binned_frames[bin_idx]) ** 2
        for i in range(n_bins):
            if count_per_bin[i] > 1:
                self.noise_estimate[i] /= (count_per_bin[i] - 1)
            else:
                self.noise_estimate[i] = np.ones_like(self.noise_estimate[i]) * Config.Gating.MLE_MIN_NOISE

        logger.info("MLE Estimator model built. Ready for estimation.")
        # Print out stats about binned frames
        for i in range(n_bins):
            logger.info(f"Bin {i}: Count={count_per_bin[i]}, Mean Pixel Value={np.mean(self.binned_frames[i]):.2f}, Mean Noise={np.mean(self.noise_estimate[i]):.2f}")
        
        


        self._ready = True
        self.frame_history = [] # Clear history to save memory after build

    def estimate(self, frame):
        scores = chi_sq(frame, self.binned_frames, self.noise_estimate)
        n_bins = len(scores)
        best_idx = np.argmin(scores)

        y = np.array([
            scores[(best_idx - 1) % n_bins], 
            scores[best_idx], 
            scores[(best_idx + 1) % n_bins]
        ])

        denom = (y[0] - 2 * y[1] + y[2])
        if denom > 0:
            offset = 0.5 * (y[0] - y[2]) / denom
            score = denom 
        else:
            offset, score = 0.0, 0.0

        # Do a parabolic fit around the best bin to get a sub-bin phase estimate
        fit_points = Config.Gating.MLE_FIT_POINTS
        from numpy.polynomial import Polynomial
        x = np.arange(-fit_points, fit_points + 1)
        y_fit = scores[(best_idx - fit_points) % n_bins : (best_idx + fit_points + 1) % n_bins]
        if len(y_fit) < 2 * fit_points + 1:
            # Handle wrap-around case
            y_fit = np.concatenate((scores[best_idx - fit_points:], scores[:best_idx + fit_points + 1 - n_bins]))
        p = Polynomial.fit(x, y_fit, 2)
        vertex = -p.coef[1] / (2 * p.coef[2]) if p.coef[2] != 0 else 0
        offset += vertex

        print(f"MLE Estimate: Best Bin={best_idx}, Offset={offset:.2f}, Score={score:.2f}")

        phase_radians = ((best_idx + offset) % n_bins / n_bins) * TWO_PI
        return phase_radians, score
    
class PhaseManager:
    def __init__(self):
        self.sad = SADEstimator()
        self.mle = MLEEstimator()

    def update(self, frame, timestamp):
        source = Config.Gating.PHASE_SOURCE
        log_all = Config.Gating.LOG_ALL
        
        mle_required = (source == "MLE" or log_all)
        results = {}
        status = "UNKNOWN"

        if not self.sad.is_ready():
            status = self.sad.add_sample(frame, timestamp=timestamp)
        
        if self.sad.is_ready():
            mle_needs_bootstrap = (mle_required and not self.mle.is_ready())

            sad_calculation_needed = (source == "SAD" or log_all or mle_needs_bootstrap)
            
            if sad_calculation_needed:
                phase_sad, score_sad = self.sad.estimate(frame)
                
                if source == "SAD" or log_all:
                    results["sad"] = {"phase": phase_sad, "score": score_sad}
                
                if mle_needs_bootstrap:
                    status = self.mle.add_sample(frame, phase=phase_sad)

        if self.mle.is_ready() and mle_required:
            phase_mle, score_mle = self.mle.estimate(frame)
            results["mle"] = {"phase": phase_mle, "score": score_mle}
            status = "READY"

        results["status"] = status
        return results