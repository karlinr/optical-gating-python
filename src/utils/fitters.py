import numpy as np
from loguru import logger
from scipy.optimize import minimize_scalar

def v_fitting(y_1, y_2, y_3):
    diff_left = y_1 - y_2
    diff_right = y_3 - y_2
    denominator = max(diff_left, diff_right)
    if denominator <= 0:
        logger.warning(f"V-fitting failed: flat or non-V shape encountered (y1={y_1}, y2={y_2}, y3={y_3}). Defaulting offset to 0.0.")
        return 0.0, float(y_2)
    x = 0.5 * (y_1 - y_3) / denominator
    y = y_2 - abs(x) * denominator
    x = np.clip(x, -0.5, 0.5)
    return x, y

def fit_v_3p(scores, best_idx):
    indices = np.arange(best_idx - 1, best_idx + 2)
    y_fit = np.take(scores, indices, mode='wrap')
    y_1, y_2, y_3 = y_fit[0], y_fit[1], y_fit[2]

    diff_left = y_1 - y_2
    diff_right = y_3 - y_2
    denominator = max(diff_left, diff_right)
    
    if denominator <= 0:
        logger.warning(f"V-3P fitting failed: flat or non-V shape encountered (y1={y_1}, y2={y_2}, y3={y_3}). Defaulting offset to 0.0.")
        return 0.0, float(y_2), None
    
    vertex_offset = 0.5 * (y_1 - y_3) / denominator
    minimized_score = y_2 - abs(vertex_offset) * denominator
    vertex_offset = np.clip(vertex_offset, -0.5, 0.5)
    
    return float(vertex_offset), float(minimized_score), None

def fit_v_np(scores, best_idx, fit_points):
    indices = np.arange(best_idx - fit_points, best_idx + fit_points + 1)
    y_fit = np.take(scores, indices, mode="wrap")
    x_fit = np.arange(-fit_points, fit_points + 1)

    def v_objective(x0):
        z = np.abs(x_fit - x0)
        z_m, y_m = np.mean(z), np.mean(y_fit)
        num = np.sum((z - z_m) * (y_fit - y_m))
        den = np.sum((z - z_m) ** 2)
        if den < 1e-12:
            return np.sum((y_fit - y_m) ** 2)
        m = num / den
        if m < 0: 
            return float('inf')
        return np.sum((y_fit - (m * z + (y_m - m * z_m))) ** 2)

    res = minimize_scalar(v_objective, bounds=(-fit_points, fit_points), method="bounded")

    if res.success:
        vertex_offset = res.x
        z = np.abs(x_fit - vertex_offset)
        z_m, y_m = np.mean(z), np.mean(y_fit)
        den = np.sum((z - z_m) ** 2)
        minimized_score = float(y_m)
        if den > 1e-12:
            m = np.sum((z - z_m) * (y_fit - y_m)) / den
            if m > 0: 
                minimized_score = y_m - m * z_m
        return float(vertex_offset), float(minimized_score), None
    else:
        logger.warning(f"V-NP fitting optimization failed: {res.message}. Defaulting offset to 0.0.")
        return 0.0, float(np.mean(y_fit)), None


def fit_u_3p(scores, best_idx):
    indices = np.arange(best_idx - 1, best_idx + 2)
    y_fit = np.take(scores, indices, mode='wrap')
    y_1, y_2, y_3 = y_fit[0], y_fit[1], y_fit[2]

    a = 0.5 * (y_1 + y_3 - 2 * y_2)
    b = 0.5 * (y_3 - y_1)
    c = y_2

    if a <= 0:
        logger.warning(f"U-3P fitting failed: Parabola is not convex (a={a}). Defaulting offset to 0.0.")
        return 0.0, float(y_2), float('inf')

    vertex_offset = -b / (2 * a)
    minimized_score = a * (vertex_offset ** 2) + b * vertex_offset + c
    uncertainty = np.sqrt(1 / a)

    if minimized_score < 0:
        logger.warning(f"U-3P minima is negative ({minimized_score:.2f}), setting to {y_2:.2f}.")
        minimized_score = y_2

    vertex_offset = np.clip(vertex_offset, -0.5, 0.5)
    return float(vertex_offset), float(minimized_score), float(uncertainty)


def fit_u_np(scores, best_idx, fit_points):
    x = np.arange(-fit_points, fit_points + 1)
    indices = np.arange(best_idx - fit_points, best_idx + fit_points + 1)
    y_fit = np.take(scores, indices, mode="wrap")

    try:
        a, b, c = np.polyfit(x, y_fit, 2)
        vertex_offset = -b / (2 * a)
        minimized_score = a * (vertex_offset ** 2) + b * vertex_offset + c

        if minimized_score < 0:
            logger.warning(f"U-NP minima is negative ({minimized_score:.2f}), setting to {scores[best_idx]:.2f}.")
            minimized_score = scores[best_idx]
    except (np.linalg.LinAlgError, TypeError):
        vertex_offset = 0.0
        minimized_score = float(scores[best_idx])
        a = 1.0
        logger.warning("U-NP parabola fitting failed. Defaulting vertex offset to 0.")

    uncertainty = np.sqrt(1 / a) if a > 0 else float('inf')
    if a <= 0:
        logger.warning("Parabola is not convex. Uncertainty set to infinity.")

    return float(vertex_offset), float(minimized_score), float(uncertainty)

def fit_poly_np(scores, best_idx, fit_points, poly_degree=2):
    x = np.arange(-fit_points, fit_points + 1)
    indices = np.arange(best_idx - fit_points, best_idx + fit_points + 1)
    y_fit = np.take(scores, indices, mode="wrap")
    
    try:
        coeffs = np.polyfit(x, y_fit, poly_degree)
        poly = np.poly1d(coeffs)
        res = minimize_scalar(poly, bounds=(-fit_points, fit_points), method='bounded')
        
        if res.success:
            vertex_offset = res.x
            minimized_score = res.fun
            deriv2 = np.polyder(poly, 2)
            a = deriv2(vertex_offset) / 2.0
            uncertainty = np.sqrt(1 / a) if a > 0 else float('inf')
            return float(vertex_offset), float(minimized_score), float(uncertainty)
    except Exception as e:
        logger.warning(f"POLY-NP fitting failed: {e}. Defaulting to discrete minimum.")
        
    return 0.0, float(scores[best_idx]), float('inf')

def fit_minima(scores, best_idx):
    return 0.0, float(scores[best_idx]), None

def interpolate_minimum(scores, best_idx, fitter_type, fit_points=1, **kwargs):
    if fitter_type == "V_3P":
        return fit_v_3p(scores, best_idx)
    elif fitter_type == "V_NP":
        return fit_v_np(scores, best_idx, fit_points)
    elif fitter_type == "U_3P":
        return fit_u_3p(scores, best_idx)
    elif fitter_type == "U_NP":
        return fit_u_np(scores, best_idx, fit_points)
    elif fitter_type == "POLY_NP":
        poly_degree = kwargs.get("poly_degree", 2)
        return fit_poly_np(scores, best_idx, fit_points, poly_degree)
    elif fitter_type == "MINIMA":
        return fit_minima(scores, best_idx)
    else:
        logger.error(f"Unknown fitter type: {fitter_type}.")
        raise ValueError(f"Unknown fitter type: {fitter_type}.")

def estimate_phase_from_scores(scores, best_idx, fitter_type, fit_points=1, **kwargs):
    reference_period = kwargs.get("reference_period")
    if reference_period is None:
        reference_period = len(scores)

    idx_offset = kwargs.get("idx_offset", 0)

    vertex_offset, minimized_score, uncertainty = interpolate_minimum(
        scores, best_idx, fitter_type, fit_points, **kwargs
    )
    
    adjusted_index = best_idx + vertex_offset - idx_offset
    phase_radians = (adjusted_index % reference_period / reference_period) * 2 * np.pi

    return {
        "phase": phase_radians,
        "vertex_offset": vertex_offset,
        "minimized_score": minimized_score,
        "uncertainty": uncertainty
    }