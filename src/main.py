import sys
from datetime import datetime
from loguru import logger
from app.data_manager import data_manager
import os
import numpy as np
import pickle

from interfaces.system import SystemController
from app.config import Config

from logic.phase_estimator import PhaseManager
from logic.predictors.base import predictor_registry
from logic.trigger_decider import TriggerDecider

import matplotlib.pyplot as plt

storage_path = Config.ExperimentConfig.EXPERIMENT_DATA_PATH
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
storage_path = f"{storage_path}/run_{timestamp_str}"

logger.remove()
logger.add(sys.stderr, level=Config.ExperimentConfig.LOGGING_LEVEL, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", enqueue=True)
logger.add(f"{storage_path}/logs/experiment.log", rotation="10 MB", level=Config.ExperimentConfig.LOGGING_LEVEL, retention="10 days", enqueue=True)

def setup_hardware(controller):
    bf_test_frame, fl_test_frame = controller.connect_all()
    logger.info("All hardware components connected successfully.")

    controller.synchronise_camera()
    controller.setup_cameras_for_experiment()
    controller.setup_timing_box_for_experiment()

    logger.success("Hardware setup for experiment complete. Ready to run.")
    return bf_test_frame.shape, fl_test_frame.shape

def initialise_metrics():
    """Initialises the structured session tracking metrics dictionary."""
    return {
        "timestamps": [],
        "framerates": [],
        "phase_results": [],
        "prediction_results": [],
        "committed_triggers": [],
    }

def run_gated_acquisition_loop(controller, phase_manager, phase_predictor, trigger_controller, metrics, iterations):
    for i in range(iterations):
        frame, timestamp, metadata = controller.get_latest_bf_frame()
        if Config.ExperimentConfig.SAVE_BRIGHTFIELD_FRAMES:
            data_manager.save("brightfield", frame.copy(), chunk_size=Config.ExperimentConfig.BRIGHTFIELD_CHUNK_SIZE)

        phase_results = phase_manager.update(frame, timestamp=timestamp)
        active = phase_results.get("ACTIVE", {})

        metrics["timestamps"].append(timestamp)
        metrics["framerates"].append(metadata.get("framerate"))
        metrics["phase_results"].append(phase_results)

        predicted_time_rel = None
        if active.get("status") == "READY":
            current_phase = active["phase"]
            target_phase = active["target_phase"]
            barrier_phase = active["barrier_phase"]
            active_metrics = active.get("metrics", {})

            logger.debug(f"Frame {i}: Current Phase={current_phase:.2f}, Target Phase={target_phase:.2f}, Barrier Phase={barrier_phase:.2f}")
            
            phase_predictor.update_phase(current_phase, timestamp, **active_metrics)
            predicted_time_rel, pred_metadata = phase_predictor.predict_target_time(target_phase, barrier_phase=barrier_phase, **active_metrics)

            if predicted_time_rel is not None:
                est_period = pred_metadata["est_period"]
                absolute_predicted_time = timestamp + predicted_time_rel

                fire_signal, relative_wait = trigger_controller.evaluate_trigger(timestamp, absolute_predicted_time, est_period)

                if fire_signal:
                    exact_hardware_target = timestamp + relative_wait
                    logger.info(f"Scheduling fluorescence trigger at absolute time {exact_hardware_target}...")
                    
                    box_time, response = controller.trigger_fl_frame(exact_hardware_target)
                    if response == 1:
                        logger.success("Fluorescence trigger successfully committed to hardware.")
                        metrics["committed_triggers"].append((timestamp, exact_hardware_target))
                        
                        def async_fluorescence_save(target = exact_hardware_target):
                            try:
                                fl_frame, fl_timestamp, fl_metadata = controller.get_latest_fl_frame()
                                if Config.ExperimentConfig.SAVE_FLUORESCENCE_FRAMES:
                                    data_manager.save("fluorescence", fl_frame, chunk_size=Config.ExperimentConfig.FLUORESCENCE_CHUNK_SIZE)
                                logger.success(f"Asynchronously saved FL frame for target time {target:.4f}")
                            except Exception as e:
                                logger.error(f"Background fluorescence pipeline failed: {e}")

                        data_manager.submit_task(async_fluorescence_save)
                    else:
                        logger.error(f"Timing Box rejected trigger target {exact_hardware_target} (Already passed).")
                        trigger_controller.handle_hardware_rejection(timestamp, est_period)
                        
        if predicted_time_rel is not None:
            metrics["prediction_results"].append((predicted_time_rel, pred_metadata))
        else:
            metrics["prediction_results"].append(None)


def plot_metrics(metrics):
    timestamps = metrics["timestamps"]
    
    # Identify all individual estimators present in the results (excluding the 'ACTIVE' metadata block)
    est_names = sorted(list(set(name for r in metrics["phase_results"] for name in r if name != "ACTIVE")))

    active_res = [r.get("ACTIVE", {}) for r in metrics["phase_results"]]
    active_phases = [a.get("phase") for a in active_res]

    periods = [p[1]["est_period"] if p else None for p in metrics["prediction_results"]]
    lookaheads = [p[0] if p else None for p in metrics["prediction_results"]]
    k_phases = [p[1].get("phase_estimate") if p else None for p in metrics["prediction_results"]]
    k_velocities = [p[1].get("phase_velocity_estimate") if p else None for p in metrics["prediction_results"]]

    # 5x2 layout grid to track alignment metrics, tracking states, and stability distributions
    fig, axs = plt.subplots(5, 2, figsize=(14, 22))
    axes = axs.flatten()
    
    def try_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    # 0. Camera Framerate
    axes[0].plot(timestamps, metrics["framerates"])
    axes[0].set_title("Camera Framerate (fps)")
    
    # 1. Model Fit / Alignment Scores
    for name in est_names:
        chi_squares = [r.get(name, {}).get("metrics", {}).get("reduced_chi_squared") for r in metrics["phase_results"]]
        if any(c is not None for c in chi_squares):
            axes[1].plot(timestamps, chi_squares, label=f"{name} Reduced Chi2")
    axes[1].set_title("Model Fit / Alignment Scores")
    try_legend(axes[1])

    # 2. Phase Estimates
    for name in est_names:
        phases = [r.get(name, {}).get("phase") for r in metrics["phase_results"]]
        axes[2].plot(timestamps, phases, label=f"{name} Estimate", alpha=0.4, linestyle=":")
        
        uncertainties = [r.get(name, {}).get("metrics", {}).get("uncertainty_estimate") for r in metrics["phase_results"]]
        if any(u is not None for u in uncertainties):
            p_arr = np.array([p if p is not None else np.nan for p in phases])
            u_arr = np.array([u if u is not None else 0.0 for u in uncertainties])
            
            axes[2].errorbar(
                timestamps, p_arr, yerr=u_arr, fmt='none', 
                elinewidth=1, capsize=2, alpha=0.4, 
                label=f"{name} $\sigma$ Bound"
            )
            
    axes[2].plot(timestamps, active_phases, label="Active Phase", color="black", linewidth=1.5)
    if any(p is not None for p in k_phases):
        axes[2].plot(timestamps, k_phases, label="Kalman Phase", linestyle="--")
    axes[2].set_title("Phase Estimates Over Time")
    try_legend(axes[2])

    # 3. Kalman Phase Velocity
    if any(p is not None for p in k_velocities):
        axes[3].plot(timestamps, k_velocities)
    axes[3].set_title("Kalman Phase Velocity")

    # 4. Cardiac Period
    axes[4].plot(timestamps, periods)
    axes[4].set_title("Estimated Cardiac Period (s)")

    # 5. Lookahead and Triggers
    pred_pts = [(t, t + l) for t, l in zip(timestamps, lookaheads) if l is not None]
    if pred_pts:
        px, py = zip(*pred_pts)
        axes[5].scatter(px, py, label="Predicted Target Time", s=1)
    if metrics["committed_triggers"]:
        tx, ty = zip(*metrics["committed_triggers"])
        axes[5].scatter(tx, ty, label="Trigger", marker="x", color="red")
    axes[5].set_title("Gated Lookahead and Hardware Trigger Commitments")
    try_legend(axes[5])

    # 6. Drift X/Y
    for name in est_names:
        drift_xs = [r.get(name, {}).get("metrics", {}).get("drift_x") for r in metrics["phase_results"]]
        drift_ys = [r.get(name, {}).get("metrics", {}).get("drift_y") for r in metrics["phase_results"]]
        if any(d is not None for d in drift_xs):
            axes[6].plot(timestamps, drift_xs, label=f"{name} Drift X")
        if any(d is not None for d in drift_ys):
            axes[6].plot(timestamps, drift_ys, label=f"{name} Drift Y")
    axes[6].set_title("Drift (pixels)")
    try_legend(axes[6])

    # 7. Delta Phase vs Phase
    for name in est_names:
        valid_phases = []
        valid_uncertainties = []
        
        for r in metrics.get("phase_results", []):
            if r and isinstance(r, dict):
                est_res = r.get(name)
                if est_res and isinstance(est_res, dict):
                    phase = est_res.get("phase")
                    m = est_res.get("metrics", {})
                    unc = m.get("uncertainty_estimate")
                    if phase is not None:
                        valid_phases.append(phase)
                        valid_uncertainties.append(unc if unc is not None else 0.0)
                        
        if len(valid_phases) > 1:
            valid_phases = np.array(valid_phases)
            valid_uncertainties = np.array(valid_uncertainties)
            dp = np.diff(np.unwrap(valid_phases))
            x_phase = np.mod(valid_phases[:-1], 2 * np.pi)
            axes[7].scatter(x_phase, dp, s=5, alpha=0.5, label=f"{name} Delta Phase")

            if np.any(valid_uncertainties > 0):
                sigma_dp = np.sqrt(valid_uncertainties[:-1]**2 + valid_uncertainties[1:]**2)
                x_err = valid_uncertainties[:-1]
                
                axes[7].errorbar(
                    x_phase, dp, 
                    yerr=sigma_dp, xerr=x_err, 
                    fmt='none', elinewidth=0.8, capsize=1.5, alpha=0.4, ecolor='gray'
                )

    axes[7].set_title("Delta Phase vs Phase")
    axes[7].set_xlabel("Phase (radians)")
    axes[7].set_ylabel("Delta Phase (radians)")
    axes[7].set_xlim(0, 2 * np.pi)
    try_legend(axes[7])

    # 8. Categorical State Probabilities (Outliers and Best Matching Phase Bin)
    for name in est_names:
        p_anon = [r.get(name, {}).get("probabilities", {}).get("unknown_anomaly") for r in metrics["phase_results"]]
        
        # Extract the probability associated with whichever discrete phase bin was selected as the winner
        p_best = []
        for r in metrics["phase_results"]:
            est_res = r.get(name, {})
            p_bins = est_res.get("probabilities", {}).get("phase_bins")
            b_idx = est_res.get("metrics", {}).get("best_index")
            if p_bins is not None and b_idx is not None:
                p_best.append(p_bins[b_idx])
            else:
                p_best.append(None)
            
        if any(p is not None for p in p_anon):
            axes[8].plot(timestamps, p_anon, label=f"{name} Outliers (Anomaly)", color="crimson", alpha=0.8)
        if any(p is not None for p in p_best):
            axes[8].plot(timestamps, p_best, label=f"{name} Best Bin Probability", color="forestgreen", alpha=0.8)
            
    axes[8].set_title("System State Probabilities")
    axes[8].set_xlabel("Time (s)")
    axes[8].set_ylabel("Probability")
    axes[8].set_ylim(-0.05, 1.05)
    axes[8].grid(True, alpha=0.3)
    try_legend(axes[8])

    # 9. Individual Phase Bin Probabilities Waterfall Heatmap
    for name in est_names:
        n_bins_detected = None
        for r in metrics["phase_results"]:
            p_bins = r.get(name, {}).get("probabilities", {}).get("phase_bins")
            if p_bins is not None:
                n_bins_detected = len(p_bins)
                break
        
        if n_bins_detected is not None:
            bin_matrix = np.full((n_bins_detected, len(timestamps)), np.nan)
            for f_idx, r in enumerate(metrics["phase_results"]):
                p_bins = r.get(name, {}).get("probabilities", {}).get("phase_bins")
                if p_bins is not None:
                    bin_matrix[:, f_idx] = p_bins
            
            im = axes[9].imshow(
                bin_matrix, aspect='auto', origin='lower',
                extent=[0, len(timestamps), 0, n_bins_detected],
                cmap='viridis', vmin=0.0, vmax=1.0
            )
            fig.colorbar(im, ax=axes[9], orientation='vertical', label='Bin Probability')
            axes[9].set_ylabel("Bin Index")
            axes[9].set_xlabel("Frame Index")
            break

    axes[9].set_title("Phase Bin Probability Distribution Waterfall")

    plt.tight_layout()
    plt.savefig(os.path.join(storage_path, "acquisition_metrics.png"))
    plt.show()

def plot_peak_locking_diagnostics(metrics):
    """
    Generates diagnostic plots to detect peak locking in temporal sub-bin 
    phase estimation and spatial drift tracking.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    est_names = sorted(list(set(name for r in metrics["phase_results"] for name in r if name != "ACTIVE")))
    if not est_names:
        logger.warning("No estimator metrics found for peak locking analysis.")
        return

    fig, axs = plt.subplots(len(est_names), 3, figsize=(18, 5 * len(est_names)))
    
    if len(est_names) == 1:
        axs = np.expand_dims(axs, axis=0)

    for idx, name in enumerate(est_names):
        vertex_offsets = []
        drift_xs = []
        drift_ys = []
        
        for r in metrics.get("phase_results", []):
            if r and isinstance(r, dict):
                est_res = r.get(name)
                if est_res and isinstance(est_res, dict):
                    m = est_res.get("metrics", {})
                    v_off = m.get("vertex_offset")
                    dx = m.get("drift_x")
                    dy = m.get("drift_y")
                    
                    if v_off is not None:
                        vertex_offsets.append(v_off)
                    if dx is not None:
                        drift_xs.append(dx)
                    if dy is not None:
                        drift_ys.append(dy)

        if vertex_offsets:
            axs[idx, 0].hist(vertex_offsets, bins=50, range=(-1.0, 1.0), alpha=0.75, color='royalblue', edgecolor='black')
            axs[idx, 0].axvline(0.0, color='crimson', linestyle='--', linewidth=1.5)
            axs[idx, 0].set_title(f"{name} - Phase Sub-bin Offsets")
            axs[idx, 0].set_xlabel("Vertex Offset (bins)")
            axs[idx, 0].set_ylabel("Count")
            axs[idx, 0].grid(True, alpha=0.3)
            
        if drift_xs and drift_ys:
            frac_x = np.array(drift_xs) - np.round(drift_xs)
            frac_y = np.array(drift_ys) - np.round(drift_ys)
            
            axs[idx, 1].hist(frac_x, bins=50, range=(-0.5, 0.5), alpha=0.5, label='Drift X Frac', color='teal')
            axs[idx, 1].hist(frac_y, bins=50, range=(-0.5, 0.5), alpha=0.5, label='Drift Y Frac', color='darkorange')
            axs[idx, 1].axvline(0.0, color='crimson', linestyle='--', linewidth=1.5)
            axs[idx, 1].set_title(f"{name} - Fractional Spatial Drift")
            axs[idx, 1].set_xlabel("Fractional Pixel Error")
            axs[idx, 1].set_ylabel("Count")
            axs[idx, 1].legend()
            axs[idx, 1].grid(True, alpha=0.3)
            
            axs[idx, 2].scatter(frac_x, frac_y, alpha=0.2, s=6, color='purple')
            axs[idx, 2].axhline(0.0, color='black', linestyle=':', alpha=0.4)
            axs[idx, 2].axvline(0.0, color='black', linestyle=':', alpha=0.4)
            axs[idx, 2].set_title(f"{name} - 2D Fractional Drift")
            axs[idx, 2].set_xlabel("Fractional X")
            axs[idx, 2].set_ylabel("Fractional Y")
            axs[idx, 2].set_xlim(-0.5, 0.5)
            axs[idx, 2].set_ylim(-0.5, 0.5)
            axs[idx, 2].set_aspect('equal')

    plt.tight_layout()
    try:
        import os
        plt.savefig(os.path.join(storage_path, "peak_locking_diagnostics.png"), dpi=150)
    except Exception as e:
        logger.error(f"Could not save peak locking plot: {e}")
    plt.show()

def main():
    with SystemController() as controller:
        metrics = initialise_metrics()
        setup_hardware(controller)
        data_manager.configure(storage_path)

        try:
            phase_manager = PhaseManager()
            pred_method = Config.Gating.PREDICTION_METHOD
            if pred_method in predictor_registry:
                phase_predictor = predictor_registry[pred_method]()
            else:
                raise ValueError(f"Unsupported prediction method: {pred_method}")
            
            trigger_controller = TriggerDecider()
            run_gated_acquisition_loop(controller, phase_manager, phase_predictor, trigger_controller, metrics, iterations=Config.ExperimentConfig.ITERATIONS)


            metrics_save_path = os.path.join(storage_path, "metrics.pkl")
            with open(metrics_save_path, "wb") as f:
                pickle.dump(metrics, f)
            logger.info(f"Metrics saved to {metrics_save_path}")

            logger.info("Acquisition loop finished. Rendering metrics...")
            plot_metrics(metrics)
            plot_peak_locking_diagnostics(metrics)
        finally:
            data_manager.close()

if __name__ == "__main__":
    main()