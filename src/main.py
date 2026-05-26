import sys
from datetime import datetime
from loguru import logger
from app.data_manager import data_manager
import os
import numpy as np

from interfaces.system import SystemController
from app.config import Config

from logic.phase_estimator import PhaseManager
from logic.phase_predictor import BarrierPredictor, KalmanPredictor, TriggerDecider

import matplotlib.pyplot as plt

storage_path = Config.ExperimentConfig.EXPERIMENT_DATA_PATH
# Add timestamp to storage path for this run using time
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
        "sad_phases": [],
        "mle_phases": [],
        "active_phases": [],
        "est_periods": [],
        "predicted_lookaheads": [],
        "committed_triggers": []  # List of tuples: (timestamp_issued, absolute_target_time)
    }

def run_gated_acquisition_loop(controller, phase_manager, phase_predictor, trigger_controller, metrics, iterations):
    for i in range(iterations):
        # Grab latest brightfield frame, timestamp, and instant framerate
        frame, timestamp, framerate = controller.get_latest_bf_frame()
        data_manager.save("brightfield", frame.copy(), chunk_size=Config.ExperimentConfig.BRIGHTFIELD_CHUNK_SIZE)

        # Update our phase estimate based on the new frame
        phase_results = phase_manager.update(frame, timestamp=timestamp)
        active = phase_results.get("ACTIVE", {})

        metrics["timestamps"].append(timestamp)
        metrics["framerates"].append(framerate)
        metrics["sad_phases"].append(phase_results.get("SAD", {}).get("phase", None))
        metrics["mle_phases"].append(phase_results.get("MLE", {}).get("phase", None))
        metrics["active_phases"].append(active.get("phase", None))

        if active.get("status") == "READY":
            current_phase = active["phase"]
            target_phase = active["target_phase"]
            barrier_phase = active["barrier_phase"]
            sad_metrics = phase_results.get("SAD", {}).get("metrics", {})
            best_index = sad_metrics.get("best_index", 0)
            reference_period = sad_metrics.get("reference_period", 1)
            
            uncertainty_estimate = active.get("metrics", {}).get("uncertainty_estimate", None)
            phase_predictor.update_phase(current_phase, timestamp, uncertainty_estimate=uncertainty_estimate)
            prediction_results = phase_predictor.predict_target_time(
                target_phase, 
                barrier_phase=barrier_phase, 
                best_index=best_index, 
                reference_period=reference_period
            )

            if prediction_results is not None:
                predicted_time_rel = prediction_results["predicted_time_rel"]
                est_period = prediction_results["metrics"]["est_period"]

                metrics["est_periods"].append(est_period)
                metrics["predicted_lookaheads"].append(predicted_time_rel)
                
                # Translate the relative lookahead delay into an absolute timeline target
                absolute_predicted_time = timestamp + predicted_time_rel

                # Evaluate tracking thresholds and check half-cycle lockout guards
                fire_signal, relative_wait = trigger_controller.evaluate_trigger(
                    timestamp, absolute_predicted_time, est_period
                )

                if fire_signal:
                    exact_hardware_target = timestamp + relative_wait
                    logger.info(f"Scheduling fluorescence trigger at absolute time {exact_hardware_target}...")
                    
                    box_time, response = controller.trigger_fl_frame(exact_hardware_target)
                    if response == 1:
                        logger.success("Fluorescence trigger successfully committed to hardware.")
                        metrics["committed_triggers"].append((timestamp, exact_hardware_target))
                        
                        def async_fluorescence_save():
                            try:
                                fl_frame, fl_timestamp = controller.get_latest_fl_frame()
                                data_manager.save("fluorescence", fl_frame, chunk_size=Config.ExperimentConfig.FLUORESCENCE_CHUNK_SIZE)
                                logger.success(f"Asynchronously saved FL frame for target time {exact_hardware_target:.4f}")
                            except Exception as e:
                                logger.error(f"Background fluorescence pipeline failed: {e}")

                        # Dispatch the combined pipeline function to the background threads
                        data_manager.submit_task(async_fluorescence_save)
                    else:
                        # Handle hardware collision if the target deadline already expired
                        logger.error(f"Timing Box rejected trigger target {exact_hardware_target} (Already passed).")
                        trigger_controller.handle_hardware_rejection(timestamp, est_period)
            else:
                metrics["est_periods"].append(None)
                metrics["predicted_lookaheads"].append(None)
        else:
            metrics["est_periods"].append(None)
            metrics["predicted_lookaheads"].append(None)


def plot_metrics(metrics):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(metrics["timestamps"], metrics["framerates"], label="Framerate")
    plt.xlabel("Time (s)")
    plt.ylabel("Framerate (fps)")
    plt.title("Camera Framerate Over Time")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(metrics["timestamps"], metrics["sad_phases"], label="SAD Phase")
    plt.plot(metrics["timestamps"], metrics["mle_phases"], label="MLE Phase")
    plt.plot(metrics["timestamps"], metrics["active_phases"], label="Active Phase")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (degrees)")
    plt.title("Phase Estimates Over Time")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(metrics["timestamps"], metrics["est_periods"], label="Estimated Period")
    plt.xlabel("Time (s)")
    plt.ylabel("Period (s)")
    plt.title("Estimated Cardiac Period Over Time")
    plt.legend()

    plt.subplot(2, 2, 4)
    predicted_times = [t for t in metrics["predicted_lookaheads"] if t is not None]
    predicted_timestamps = [metrics["timestamps"][i] for i in range(len(metrics["predicted_lookaheads"])) if metrics["predicted_lookaheads"][i] is not None]
    plt.scatter(predicted_timestamps, np.array(predicted_timestamps) + np.array(predicted_times), label="Predicted Lookahead", color='orange')
    
    # Plot commited timestamps and triggers
    committed_timestamps = [t[0] for t in metrics["committed_triggers"]]
    committed_targets = [t[1] for t in metrics["committed_triggers"]]
    plt.scatter(committed_timestamps, committed_targets, label="Committed Trigger", color='red')
    
    
    plt.xlabel("Time (s)")
    plt.ylabel("Time to Target (s)")
    plt.title("Predicted Lookahead and Committed Trigger Times")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    with SystemController() as controller:
        metrics = initialize_metrics()
        setup_hardware(controller)

        data_manager.configure(storage_path)

        phase_manager = PhaseManager()
        if Config.Gating.PREDICTION_METHOD == "KALMAN":
            phase_predictor = KalmanPredictor()
        elif Config.Gating.PREDICTION_METHOD == "BARRIER":
            phase_predictor = BarrierPredictor()
        else:
            raise ValueError(f"Unsupported prediction method: {Config.Gating.PREDICTION_METHOD}")
        trigger_controller = TriggerDecider()
        
        run_gated_acquisition_loop(controller, phase_manager, phase_predictor, trigger_controller, metrics, iterations=5000)

        plot_metrics(metrics)

        logger.info("Acquisition loop finished. Rendering metrics...")

if __name__ == "__main__":
    main()