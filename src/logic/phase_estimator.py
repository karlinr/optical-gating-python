import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List, Any

from logic.utils import v_fitting, chi_sq, sad_with_references
from app.config import Config

# Constants
TWO_PI = 2 * np.pi
REFERENCE_PADDING = 2

logger = logging.getLogger("PhaseEstimator")

class PhaseEstimator(ABC):
    def __init__(self):
        self._ready = False

    def is_ready(self):
        return self._ready

    @abstractmethod
    def add_sample(self, frame, **kwargs):
        """Adds a single frame to the estimator's internal history."""
        pass

    @abstractmethod
    def estimate(self, frame):
        """Estimates the phase and returns (phase, score)."""
        pass

class SADEstimator(PhaseEstimator):
    def __init__(self):
        super().__init__()
        self.reference_frames = None
        self.reference_period = None
        self.barrier_phase = None
        self.frame_history = []
    
    def add_sample(self, frame, **kwargs):
        self.frame_history.append(frame)
        if len(self.frame_history) >= Config.Gating.REFERENCE_LENGTH and not self._ready:
            self.build_model(self.frame_history)
    
    def build_model(self, frames):
        # To be copied over from main codebase
        if (self.reference_frames is not None and 
            self.reference_period is not None and 
            self.barrier_phase is not None):
            self._ready = True

    def estimate(self, frame):
        scores = sad_with_references(frame, self.reference_frames)
        search_window = scores[REFERENCE_PADDING : -REFERENCE_PADDING]
        best_idx_in_window = np.argmin(search_window)
        best_idx_abs = best_idx_in_window + REFERENCE_PADDING

        offset, score = v_fitting(
            scores[best_idx_abs - 1], scores[best_idx_abs], scores[best_idx_abs + 1]
        )

        phase = ((best_idx_in_window + offset) / self.reference_period) * TWO_PI

        # We return score as the minima of the SAD curve although I don't expect we will use this
        # this is purely included to match the output format of the MLE estimator
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

        if len(self.frame_history) >= Config.Gating.BOOTSTRAP_FRAMES and not self._ready:
            logger.info("MLE Estimator has enough samples. Building model.")
            self.build_model()

    def build_model(self):
        # Again, needs to be copied over from my main code.
        # This will bin frames by phase and estimating pixel-wise mean and variance for each bin.
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

        phase_radians = ((best_idx + offset) % n_bins / n_bins) * TWO_PI
        return phase_radians, score
    
class PhaseManager:
    def __init__(self):
        self.sad = SADEstimator()
        self.mle = MLEEstimator()

    def update(self, frame):
        source = Config.Gating.PHASE_SOURCE
        log_all = Config.Gating.LOG_ALL
        
        mle_required = (source == "MLE" or log_all)
        results = {}

        if not self.sad.is_ready():
            self.sad.add_sample(frame)
        
        if self.sad.is_ready():
            mle_needs_bootstrap = (mle_required and not self.mle.is_ready())
            sad_calculation_needed = (source == "SAD" or log_all or mle_needs_bootstrap)
            
            if sad_calculation_needed:
                phase_sad, score_sad = self.sad.estimate(frame)
                
                if source == "SAD" or log_all:
                    results["sad"] = {"phase": phase_sad, "score": score_sad}
                
                if mle_needs_bootstrap:
                    self.mle.add_sample(frame, phase=phase_sad)

        if self.mle.is_ready() and mle_required:
            phase_mle, score_mle = self.mle.estimate(frame)
            results["mle"] = {"phase": phase_mle, "score": score_mle}

        return results