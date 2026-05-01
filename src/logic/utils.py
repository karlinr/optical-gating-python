from numba import njit, prange
import numpy as np
from loguru import logger

def v_fitting(y_1, y_2, y_3):
    """
    Fit a symmetric V-shape to three points to find the interpolated minimum.
    """
    # Calculate difference from center point
    diff_left = y_1 - y_2
    diff_right = y_3 - y_2
    
    # Identify the steepness
    denominator = max(diff_left, diff_right)
    
    # Handle non-V shapes or flat lines
    if denominator <= 0:
        return 0.0, float(y_2)
    
    # Calculate vertex coordinates
    x = 0.5 * (y_1 - y_3) / denominator
    y = y_2 - abs(x) * denominator
    
    return x, y

@njit(parallel = True)
def chi_sq(test_frame, binned_frames, noise_est):
    """
    Compute chi-squared values for a test frame against multiple binned frames with noise estimates.
    """
    # Test frmame (w,h) - 2D array
    # binned_frames (n_bins, w, h) - 3D array
    # noise_est (n_bins, w, h) - 3D array
    n_bins, w, h = binned_frames.shape

    chi_sq_terms = np.zeros(n_bins, dtype=np.float64)

    for i in prange(n_bins):
        acc = 0.0
        for x in range(w):
            for y in range(h):
                obs = test_frame[x, y]
                exp = binned_frames[i, x, y]
                var = noise_est[i, x, y]
                acc += (obs - exp) ** 2 / var
        chi_sq_terms[i] = acc

    return chi_sq_terms

@njit(parallel=True, fastmath=True)
def sad_with_references(test_frame, reference_stack):
    """
    Compute Sum of Absolute Differences (SAD) between a test frame and multiple reference frames.
    This is written as a drop in replacement for Jonathan Taylor's original SAD implementation
    """
    n_refs, w, h = reference_stack.shape

    sad_scores = np.zeros(n_refs, dtype=np.int64)

    for i in prange(n_refs):
        acc = np.int64(0)
        for x in range(w):
            for y in range(h):
                obs = np.int32(test_frame[x, y])
                ref = np.int32(reference_stack[i, x, y])
                acc += abs(obs - ref)
        sad_scores[i] = acc
        
    return sad_scores