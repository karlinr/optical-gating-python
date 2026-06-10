import numpy as np
from scipy.ndimage import shift
from app.config import Config
from loguru import logger
from logic.utils import sad_with_reference

class DriftCorrector:
    """
    New drift corrector based on original in open-optical-gating but with sub-pixel drift estimation and correction
    """
    def __init__(self):
        self.search_radius = Config.Gating.DRIFT_INITIAL_SEARCH
        self.drift_x = 0.0
        self.drift_y = 0.0

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
            logger.error(f"Drift tracking bounds ({search_margin_x}, {search_margin_y}) exceed image center. Resetting tracker.")
            self.drift_x, self.drift_y = 0.0, 0.0
            return 0.0, 0.0

        ref_center = reference[search_margin_x : w - search_margin_x, search_margin_y : h - search_margin_y]

        eval_radius = self.search_radius + 1
        grid_size = 2 * eval_radius + 1
        sad_grid = np.empty((grid_size, grid_size))
        offset = eval_radius

        for s_dx in range(-eval_radius, eval_radius + 1):
            for s_dy in range(-eval_radius, eval_radius + 1):
                dx = initial_dx + s_dx
                dy = initial_dy + s_dy

                live_slice = frame[
                    search_margin_x - dx : w - search_margin_x - dx,
                    search_margin_y - dy : h - search_margin_y - dy,
                ]
                
                sad_grid[s_dx + offset, s_dy + offset] = sad_with_reference(live_slice, ref_center)

        inner_grid = sad_grid[1:-1, 1:-1]
        min_inner_idx = np.unravel_index(np.argmin(inner_grid), inner_grid.shape)
        
        best_grid_x = min_inner_idx[0] + 1
        best_grid_y = min_inner_idx[1] + 1

        best_dx = initial_dx + (best_grid_x - offset)
        best_dy = initial_dy + (best_grid_y - offset)

        center = sad_grid[best_grid_x, best_grid_y]
        left   = sad_grid[best_grid_x - 1, best_grid_y]
        right  = sad_grid[best_grid_x + 1, best_grid_y]
        up     = sad_grid[best_grid_x, best_grid_y - 1]
        down   = sad_grid[best_grid_x, best_grid_y + 1]

        denom_x = 2.0 * (left + right - 2.0 * center)
        denom_y = 2.0 * (up + down - 2.0 * center)

        sub_dx = (left - right) / denom_x if abs(denom_x) > 1e-6 else 0.0
        sub_dy = (up - down)   / denom_y if abs(denom_y) > 1e-6 else 0.0

        self.drift_x = best_dx + np.clip(sub_dx, -1.0, 1.0)
        self.drift_y = best_dy + np.clip(sub_dy, -1.0, 1.0)

        self.search_radius = Config.Gating.DRIFT_MAX_SEARCH

        return self.drift_x, self.drift_y

    def adjust_reference_array(self, reference, drift=None):
        if not Config.Gating.DRIFT_CORRECT:
            return reference

        dx = drift[0] if drift is not None else self.drift_x
        dy = drift[1] if drift is not None else self.drift_y
        mx = int(np.ceil(abs(dx))) + 1
        my = int(np.ceil(abs(dy))) + 1

        if reference.ndim == 3:
            return reference[:, mx:-mx, my:-my]
        else:
            return reference[mx:-mx, my:-my]

    def adjust_live_frame(self, frame, drift=None):
        if not Config.Gating.DRIFT_CORRECT:
            return frame

        dx = drift[0] if drift is not None else self.drift_x
        dy = drift[1] if drift is not None else self.drift_y
        mx = int(np.ceil(abs(dx))) + 1
        my = int(np.ceil(abs(dy))) + 1

        stabilized = shift(frame, shift=(dx, dy), order=1, mode="constant", cval=0.0)
        return stabilized[mx:-mx, my:-my]