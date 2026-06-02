from loguru import logger
import numpy as np
from abc import ABC, abstractmethod

# Ensure we use the optimized utils
from logic.estimators.base import register_estimator, PhaseEstimator
from logic.utils import v_fitting, chi_sq, sad_with_references 
from app.config import Config
from app.data_manager import data_manager

@register_estimator("SAD")
class SADEstimator(PhaseEstimator):
    """
    Standard SAD-based estimator as used in original optical-gating code
    Much of this is adapted from open-optical-gating @ https://github.com/Glasgow-ICG/open-optical-gating/tree/main
    """
    dependencies = []

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

        data_manager.save("reference_sequence", self.reference_frames.copy())

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
            "target_phase": self.target_phase,
            "barrier_phase": self.barrier_phase,
            "metrics": {
                "sad_score": score,
                "best_index": best_idx - Config.Gating.NUM_EXTRA_REF_FRAMES,
                "offset": offset,
                "reference_period": self.reference_period,
                "scores": scores
            }
        }