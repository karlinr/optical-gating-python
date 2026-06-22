from numba import njit, prange, set_num_threads
import numpy as np
from loguru import logger
from app.config import Config

set_num_threads(Config.ExperimentConfig.NUM_THREADS)

@njit(parallel=True, cache=True)
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
                obs = float(test_frame[x, y])
                exp = float(binned_frames[i, x, y])
                var = float(noise_est[i, x, y])
                acc += ((obs - exp) ** 2 / var)
        chi_sq_terms[i] = acc

    return chi_sq_terms

@njit(parallel=True, fastmath=True, cache=True)
def sad_with_references(test_frame, reference_stack):
    """
    Compute Sum of Absolute Differences (SAD) between a test frame and multiple reference frames.
    This is written as a drop in replacement for Jonathan Taylor's original SAD implementation
    """
    n_refs, w, h = reference_stack.shape

    sad_scores = np.zeros(n_refs, dtype=np.float64)

    for i in prange(n_refs):
        acc = 0.0
        for x in range(w):
            for y in range(h):
                obs = float(test_frame[x, y])
                ref = float(reference_stack[i, x, y])
                acc += abs(obs - ref)
        sad_scores[i] = acc
        
    return sad_scores

@njit(cache=True, parallel=True)
def compute_sad_grid(frame, reference, initial_dx, initial_dy, eval_radius, margin_x, margin_y):
    """
    Calculate the grid of SAD values for a given frame and reference
    """
    w, h = frame.shape[:2]
    grid_size = 2 * eval_radius + 1
    sad_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    
    # Slice the reference centre
    ref_centre = reference[margin_x : w - margin_x, margin_y : h - margin_y]
    sub_w, sub_h = ref_centre.shape[:2]

    for s_dx in prange(-eval_radius, eval_radius + 1):
        for s_dy in range(-eval_radius, eval_radius + 1):
            dx = initial_dx + s_dx
            dy = initial_dy + s_dy

            # Calculate the top-left starting pixel position on the live frame
            live_start_x = margin_x - dx
            live_start_y = margin_y - dy
            
            acc = 0.0
            
            for x in range(sub_w):
                for y in range(sub_h):
                    obs = float(frame[live_start_x + x, live_start_y + y])
                    ref = float(ref_centre[x, y])
                    acc += abs(obs - ref)
            
            sad_grid[s_dx + eval_radius, s_dy + eval_radius] = acc

    return sad_grid