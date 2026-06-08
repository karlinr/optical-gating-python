import sys
from datetime import datetime
from loguru import logger
from app.data_manager import data_manager
import os
import numpy as np

from interfaces.system import SystemController
from app.config import Config

from logic.phase_estimator import PhaseManager
from logic.predictors.base import predictor_registry
from logic.estimators.base import estimator_registry
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

def initialize_metrics():
    """Initializes the structured session tracking metrics dictionary."""
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
    
    chi_squares = [r.get("ACTIVE", {}).get("metrics", {}).get("reduced_chi_squared") for r in metrics["phase_results"]]
    active_phases = [r.get("ACTIVE", {}).get("phase") for r in metrics["phase_results"]]
    periods = [p[1]["est_period"] if p else None for p in metrics["prediction_results"]]
    lookaheads = [p[0] if p else None for p in metrics["prediction_results"]]
    k_phases = [p[1].get("phase_estimate") if p else None for p in metrics["prediction_results"]]
    k_velocities = [p[1].get("phase_velocity_estimate") if p else None for p in metrics["prediction_results"]]

    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    
    axs[0, 0].plot(timestamps, metrics["framerates"], color='tab:blue')
    axs[0, 0].set_title("Camera Framerate (fps)")
    
    axs[0, 1].plot(timestamps, chi_squares, color="purple")
    axs[0, 1].set_title("Model Fit Goodness (Reduced $\chi^2$)")

    est_names = set(n for r in metrics["phase_results"] for n in r if n != "ACTIVE")
    for name in est_names:
        phases = [r.get(name, {}).get("phase") for r in metrics["phase_results"]]
        axs[1, 0].plot(timestamps, phases, label=f"{name} Estimate", alpha=0.4, linestyle=":")
    axs[1, 0].plot(timestamps, active_phases, label="Active Phase", color="black", lw=1.5)
    if any(p is not None for p in k_phases):
        axs[1, 0].plot(timestamps, k_phases, label="Kalman Phase ($X_0$)", color="darkblue", linestyle='--')
    axs[1, 0].set_title("Phase Estimates Over Time")

    if any(p is not None for p in k_velocities):
        axs[1, 1].plot(timestamps, k_velocities, color="crimson")
    axs[1, 1].set_title("Kalman Phase Velocity ($X_1$)")

    axs[2, 0].plot(timestamps, periods, color="tab:green")
    axs[2, 0].set_title("Estimated Cardiac Period (s)")

    pred_pts = [(t, t + l) for t, l in zip(timestamps, lookaheads) if l is not None]
    if pred_pts:
        px, py = zip(*pred_pts)
        axs[2, 1].scatter(px, py, label="Predicted Target Time", color="orange", s = 1)
    if metrics["committed_triggers"]:
        tx, ty = zip(*metrics["committed_triggers"])
        axs[2, 1].scatter(tx, ty, label="Trigger", color="red", marker="x")
    axs[2, 1].set_title("Gated Lookahead and Hardware Trigger Commitments")

    for ax in axs.flat:
        ax.grid(True, linestyle="--", alpha=0.4)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize='small')
            
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(storage_path, "acquisition_metrics.png"))
    plt.show()


def main():
    with SystemController() as controller:
        metrics = initialize_metrics()
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
            run_gated_acquisition_loop(controller, phase_manager, phase_predictor, trigger_controller, metrics, iterations=4000)

            logger.info("Acquisition loop finished. Rendering metrics...")
            plot_metrics(metrics)
        finally:
            data_manager.close()

if __name__ == "__main__":
    main()