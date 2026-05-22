from loguru import logger
import numpy as np
from abc import ABC, abstractmethod

# Ensure we use the optimized utils
from logic.utils import v_fitting, chi_sq, sad_with_references 
from app.config import Config

class PhaseEstimator(ABC):
    def __init__(self):
        self._ready = False

    def is_ready(self):
        return self._ready

    @abstractmethod
    def update(self, frame, **kwargs):
        """Processes a frame. Returns an estimation dict if ready, or None if initializing."""
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

    def update(self, frame, **kwargs):
        """Adds a frame and manages the history buffer based on heart rate."""
        if not self.is_ready():
            timestamp = kwargs.get("timestamp")
            if timestamp is None:
                logger.error("SADEstimator requires a timestamp in update.")
                return None
            
            self.frame_history.append((frame, timestamp))
            ref_buffer_duration = self.frame_history[-1][1] - self.frame_history[0][1]
            max_duration = 1.0 / Config.Gating.MIN_HEART_RATE_HZ

            while ref_buffer_duration > max_duration:
                self.frame_history.pop(0)
                ref_buffer_duration = self.frame_history[-1][1] - self.frame_history[0][1]

            if len(self.frame_history) > Config.Gating.MIN_PERIOD:
                self.build_model()

        if self.is_ready():
            return self.estimate(frame)

        return None

    def build_model(self):
        """Logic to establish period, extract sequence, and set barrier phase."""
        start, stop, period = self._establish_indices()
        
        if start is None:
            return False

        raw_sequence = [f[0] for f in self.frame_history[start:stop]]
        self.reference_frames = np.array(raw_sequence)
        self.reference_period = period

        target_frame, barrier_frame = self._pick_frames()

        logger.info(f"Target Frame: {target_frame:.2f}, Barrier Frame: {barrier_frame:.2f}")
        
        self.target_phase = 2 * np.pi * (target_frame / self.reference_period)
        self.barrier_phase = 2 * np.pi * (barrier_frame / self.reference_period)
        self._ready = True

        self.frame_history = []        
        logger.info(f"SAD Model Built: Period={period:.2f}, Target Phase={self.target_phase:.2f}, Barrier Phase={self.barrier_phase:.2f}")
        return True

    def _establish_indices(self):
        """Establish list indices representing a reference period."""
        if len(self.frame_history) < 2:
            return None, None, None

        frame = self.frame_history[-1][0]
        past_frames = np.array([f[0] for f in self.frame_history[:-1]])

        # Calculate Diffs
        diffs = sad_with_references(frame, past_frames)

        # Calculate Period length
        period = self._calculate_period_length(diffs)
        if period != -1:
            self.period_history.append(period)

        # Stability check: Requires 5 + 2*padding frames of history
        history_stable = len(self.period_history) >= (5 + (2 * Config.Gating.NUM_EXTRA_REF_FRAMES))

        if period != -1 and period > 6 and history_stable:
            period_to_use = self.period_history[-1 - Config.Gating.NUM_EXTRA_REF_FRAMES]
            
            if (len(self.period_history) - 1 - Config.Gating.NUM_EXTRA_REF_FRAMES) <= 0 or period_to_use <= 6:
                return None, None, None
                
            num_refs = int(period_to_use + 1) + (2 * Config.Gating.NUM_EXTRA_REF_FRAMES)
            
            start = len(past_frames) - num_refs
            stop = len(past_frames)

            return start, stop, period_to_use

        return None, None, None

    def _calculate_period_length(self, diffs):
        """Interpolated period search based on threshold factors."""
        if diffs.size < 2:
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
            elif score != 0 and (min_score == 0 or score < min_score):
                min_score = score

            if score < min_since_max:
                min_since_max = score
                delta_for_min_since_max = d

        if got:
            best_match_idx = diffs.size - delta_for_min_since_max
            
            if 0 < best_match_idx < diffs.size - 1:
                offset, _ = v_fitting(diffs[best_match_idx-1], diffs[best_match_idx], diffs[best_match_idx+1])
                return delta_for_min_since_max - offset
            
            return float(delta_for_min_since_max)
            
        return -1

    def _pick_frames(self):
        """Automatically identify target and barrier frames."""
        total_frames = len(self.reference_frames)
        stop_index = total_frames - Config.Gating.NUM_EXTRA_REF_FRAMES

        # Calculate deltas between consecutive frames in the sequence
        f1s = self.reference_frames[Config.Gating.NUM_EXTRA_REF_FRAMES : stop_index]
        f2s = self.reference_frames[Config.Gating.NUM_EXTRA_REF_FRAMES + 1 : stop_index + 1]
        deltas = np.sum(np.abs(f1s.astype(np.int32) - f2s.astype(np.int32)), axis=(1, 2))

        # Find the frame with the maximum delta (largest change)
        max_pos = np.argmax(deltas)

        if max_pos <= 0 or max_pos >= deltas.size - 1:
            offset = 0
            max_pos = 0
        else:
            offset, _ = v_fitting(-deltas[max_pos-1], -deltas[max_pos], -deltas[max_pos+1])

        # Target: Shift the phase forward by 1/3 of a period from the max delta peak, wrapped within the cycle
        target_frame = (max_pos + offset + (self.reference_period / 3.0)) % self.reference_period
        
        # Barrier: first point rising past midpoint between min and max deltas
        min_delta, max_delta = np.min(deltas), np.max(deltas)
        midpoint = (min_delta + max_delta) / 2

        start_barrier = np.argmin(deltas)
        barrier_frame = start_barrier
        
        while deltas[barrier_frame] < midpoint:
            barrier_frame = (barrier_frame + 1) % int(self.reference_period)

            if barrier_frame == start_barrier:
                break
            
        return target_frame, barrier_frame

    def estimate(self, frame):
        """Estimates the phase and returns (phase, score)."""
        scores = sad_with_references(frame, self.reference_frames)
        best_idx = np.argmin(scores[Config.Gating.NUM_EXTRA_REF_FRAMES : -Config.Gating.NUM_EXTRA_REF_FRAMES]) + Config.Gating.NUM_EXTRA_REF_FRAMES
        
        offset, score = v_fitting(scores[best_idx - 1], scores[best_idx], scores[best_idx + 1])
        
        phase = ((best_idx + offset - Config.Gating.NUM_EXTRA_REF_FRAMES) / self.reference_period) * 2 * np.pi

        logger.debug(f"SAD Estimate: Best Index={best_idx}, Offset={offset:.2f}, Score={score:.2f}")
        
        return {
            "phase": phase % (2 * np.pi),
            "metrics": {
                "sad_score": score,
                "best_index": best_idx - Config.Gating.NUM_EXTRA_REF_FRAMES,
                "offset": offset,
                "reference_period": self.reference_period,
                "sad_curve": scores
            }
        }
    
class MLEEstimator(PhaseEstimator):
    def __init__(self):
        super().__init__()
        self.binned_frames = None
        self.noise_estimate = None
        self.frame_history = []

    def update(self, frame, **kwargs):
        if not self.is_ready():
            phase = kwargs.get("phase")
            if phase is not None:
                self.frame_history.append((frame, phase))

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
            "metrics": {
                "reduced_chi_squared": reduced_chi_squared,
                "uncertainty_estimate": uncertainty_radians,
                "best_bin": best_idx,
                "vertex_offset": vertex_offset
            }
        }
    
class PhaseManager:
    def __init__(self):
        self.estimators = {
            "SAD": SADEstimator(),
            "MLE": MLEEstimator()
        }

    def update(self, frame, timestamp) -> dict:
        source = Config.Gating.PHASE_SOURCE
        
        # Determine which estimators need to run based on configuration
        to_run = set(Config.Gating.ENABLED_ESTIMATORS) | {source}
        # The MLE method needs the SAD to bootstrap
        if "MLE" in to_run and not self.estimators["MLE"].is_ready():
            to_run.add("SAD")

        # Run the updates
        outputs = {}
        if "SAD" in to_run:
            outputs["SAD"] = self.estimators["SAD"].update(frame, timestamp=timestamp)
            
        if "MLE" in to_run:
            sad_phase = outputs["SAD"]["phase"] if outputs.get("SAD") else None
            outputs["MLE"] = self.estimators["MLE"].update(frame, phase=sad_phase)

        # Get the results dictionary ready
        response = {}
        for name in Config.Gating.ENABLED_ESTIMATORS:
            response[name] = outputs.get(name) or {"phase": None, "metrics": {}}

        # Set the status
        is_ready = self.estimators[source].is_ready()
        status = "READY" if is_ready else f"{source}_COLLECTING_FRAMES"

        response["ACTIVE"] = {
            "status": status,
            "phase": outputs[source]["phase"] if is_ready else None,
            "target_phase": self.estimators["SAD"].target_phase if is_ready else None,
            "barrier_phase": self.estimators["SAD"].barrier_phase if is_ready else None,
            "metrics": outputs[source]["metrics"] if is_ready else {}
        }

        return response