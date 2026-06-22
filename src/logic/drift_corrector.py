import numpy as np
from app.config import Config
from loguru import logger
from utils.metrics import compute_sad_grid
from numba import njit, prange

# TODO: Right now every estimator method initiates and runs a seperate drift corrector
# Strictly speaking this is correct as we want to compare how each estimator performs but is slow and inneficient.
# We either need to consolidate and just use the SAD-based drift correction for all estimators (cascaded down)
# Or make this more efficient either using Jonny's SAD code or by using a faster method

@njit(cache=True, parallel=True)
def shift_frame(frame, dx, dy):
    """
    Bilinear frame-shift
    """
    idx = int(np.floor(dx))
    idy = int(np.floor(dy))
    fdx = dx - idx
    fdy = dy - idy

    r, c = frame.shape
    pad_x = max(abs(idx), 0) + 2
    pad_y = max(abs(idy), 0) + 2
    
    padded = np.zeros((r + 2 * pad_x, c + 2 * pad_y), dtype=frame.dtype)
    padded[pad_x:pad_x+r, pad_y:pad_y+c] = frame

    x0 = pad_x - idx
    y0 = pad_y - idy

    w00 = (1.0 - fdx) * (1.0 - fdy)
    w10 = fdx * (1.0 - fdy)
    w01 = (1.0 - fdx) * fdy
    w11 = fdx * fdy

    out = np.empty((r, c), dtype=np.float32)

    # Multi-threaded outer loop execution
    for i in prange(r):
        px = x0 + i
        for j in range(c):
            py = y0 + j
            out[i, j] = (w00 * padded[px, py] +
                         w10 * padded[px - 1, py] +
                         w01 * padded[px, py - 1] +
                         w11 * padded[px - 1, py - 1])
    return out

class DriftCorrector:
    """
    New drift corrector based on original in open-optical-gating but with sub-pixel drift estimation and correction
    """
    def __init__(self):
        self.search_radius = Config.Gating.DRIFT_INITIAL_SEARCH
        self.drift_x = 0.0
        self.drift_y = 0.0
        self.mx = int(np.ceil(abs(self.drift_x))) + 1
        self.my = int(np.ceil(abs(self.drift_y))) + 1

    def add_sample(self, frame, best_match=None):
        if not Config.Gating.DRIFT_CORRECT:
            return 0.0, 0.0

        if best_match is not None:
            self.update_drift(frame, best_match)
        return self.drift_x, self.drift_y

    def update_drift(self, frame, reference):
        w, h = frame.shape[:2]

        initial_dx = int(np.round(self.drift_x))
        initial_dy = int(np.round(self.drift_y))

        search_margin_x = max(abs(initial_dx), self.search_radius) + 2
        search_margin_y = max(abs(initial_dy), self.search_radius) + 2

        if search_margin_x >= min(w, h) // 2 or search_margin_y >= min(w, h) // 2:
            logger.error(f"Drift tracking bounds ({search_margin_x}, {search_margin_y}) exceed image centre. Resetting tracker.")
            self.drift_x, self.drift_y = 0.0, 0.0
            return 0.0, 0.0

        eval_radius = self.search_radius + 1
        offset = eval_radius

        sad_grid = compute_sad_grid(frame, reference, initial_dx, initial_dy, eval_radius, search_margin_x, search_margin_y)

        inner_grid = sad_grid[1:-1, 1:-1]
        min_inner_idx = np.unravel_index(np.argmin(inner_grid), inner_grid.shape)
        
        best_grid_x = min_inner_idx[0] + 1
        best_grid_y = min_inner_idx[1] + 1

        best_dx = initial_dx + (best_grid_x - offset)
        best_dy = initial_dy + (best_grid_y - offset)

        centre = sad_grid[best_grid_x, best_grid_y]
        left = sad_grid[best_grid_x - 1, best_grid_y]
        right = sad_grid[best_grid_x + 1, best_grid_y]
        up = sad_grid[best_grid_x, best_grid_y - 1]
        down = sad_grid[best_grid_x, best_grid_y + 1]

        denom_x = 2.0 * (left + right - 2.0 * centre)
        denom_y = 2.0 * (up + down - 2.0 * centre)

        sub_dx = (left - right) / denom_x if abs(denom_x) > 1e-6 else 0.0
        sub_dy = (up - down)   / denom_y if abs(denom_y) > 1e-6 else 0.0

        self.drift_x = best_dx + np.clip(sub_dx, -1.0, 1.0)
        self.drift_y = best_dy + np.clip(sub_dy, -1.0, 1.0)

        self.search_radius = Config.Gating.DRIFT_MAX_SEARCH

        self.mx = int(np.ceil(abs(self.drift_x))) + 1
        self.my = int(np.ceil(abs(self.drift_y))) + 1

        return self.drift_x, self.drift_y

    def adjust_reference_array(self, reference, drift=None):
        if not Config.Gating.DRIFT_CORRECT:
            return reference

        if reference.ndim == 3:
            return reference[:, self.mx:-self.mx, self.my:-self.my]
        else:
            return reference[self.mx:-self.mx, self.my:-self.my]

    def adjust_live_frame(self, frame, drift=None):
        if not Config.Gating.DRIFT_CORRECT:
            return frame

        dx = drift[0] if drift is not None else self.drift_x
        dy = drift[1] if drift is not None else self.drift_y

        stabilised = shift_frame(frame, dx, dy)
        return stabilised[self.mx:-self.mx, self.my:-self.my]