import sys
from time import time
from datetime import datetime
from loguru import logger
import zarr
from ome_zarr.writer import write_image
import tifffile as tf
import os
import numpy as np

from interfaces.system import SystemController
from app.config import Config
from app.state import AppState, ExperimentState

from logic.phase_estimator import PhaseManager
from logic.phase_predictor import BarrierPredictor, TriggerDecider

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

    brightfield_chunk_size = (Config.ExperimentConfig.BRIGHTFIELD_CHUNK_SIZE, 1, 1, bf_test_frame.shape[0], bf_test_frame.shape[1])
    fl_chunk_size = (Config.ExperimentConfig.FLUORESCENCE_CHUNK_SIZE, 1, 1, fl_test_frame.shape[0], fl_test_frame.shape[1])

    logger.info(f"Brightfield camera frame shape: {bf_test_frame.shape}, chunk size: {brightfield_chunk_size}")
    logger.info(f"Fluorescence camera frame shape: {fl_test_frame.shape}, chunk size: {fl_chunk_size}")

    controller.synchronise_camera()
    controller.setup_cameras_for_experiment()
    controller.setup_timing_box_for_experiment()
    logger.success("Hardware setup for experiment complete. Ready to run.")

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
    # Run the acquisition loop for a fixed number of iterations
    firing = False  
    fire_timestamp = 0.0

    # check if storage path exists and create if not
    if not os.path.exists(f"{storage_path}/fluorescence"):
        os.makedirs(f"{storage_path}/fluorescence")
        logger.info(f"Created storage directory at {storage_path}/fluorescence")

    for i in range(iterations):
        # Grab latest brightfield frame, timestamp, and instant framerate
        frame, timestamp, framerate = controller.get_latest_bf_frame()

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
            
            phase_predictor.update_phase(current_phase, timestamp)
            prediction_results = phase_predictor.predict_target_time(target_phase, barrier_phase, best_index, reference_period)



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

                if fire_signal and not firing:
                    exact_hardware_target = timestamp + relative_wait
                    logger.info(f"Scheduling fluorescence trigger at absolute time {exact_hardware_target}...")
                    
                    box_time, response = controller.trigger_fl_frame(exact_hardware_target)
                    if response == 1:
                        logger.success("Fluorescence trigger successfully committed to hardware.")
                        firing = True
                        fire_timestamp = exact_hardware_target
                        metrics["committed_triggers"].append((timestamp, exact_hardware_target))
                    else:
                        # Handle hardware collision if the target deadline already expired
                        logger.error(f"Timing Box rejected trigger target {exact_hardware_target} (Already passed).")
                        trigger_controller.handle_hardware_rejection(timestamp, est_period)
                        firing = False
            else:
                metrics["est_periods"].append(None)
                metrics["predicted_lookaheads"].append(None)
        else:
            metrics["est_periods"].append(None)
            metrics["predicted_lookaheads"].append(None)

        # If a trigger was committed save the fluorescence frame once exposure completes
        if firing:
            if timestamp > fire_timestamp:
                try:
                    fl_frame, fl_timestamp = controller.get_latest_fl_frame()
                    tf.imwrite(f"{storage_path}/fluorescence/fl_frame_{fl_timestamp}.tif", fl_frame)
                    logger.success(f"Fluorescence frame captured and saved at timestamp {fl_timestamp}")
                    firing = False
                except Exception as e:
                    logger.error(f"Failed to capture fluorescence frame: {e}")
                    firing = False


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
    
    """committed_times = [t[1] for t in metrics["committed_triggers"]]
    committed_timestamps = [t[0] for t in metrics["committed_triggers"]]
    plt.scatter(committed_timestamps, committed_times, label="Committed Trigger Times", color='red')"""
    
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

        phase_manager = PhaseManager()
        phase_predictor = BarrierPredictor()
        trigger_controller = TriggerDecider()
        
        run_gated_acquisition_loop(controller, phase_manager, phase_predictor, trigger_controller, metrics, iterations=5000)

        plot_metrics(metrics)

        logger.info("Acquisition loop finished. Rendering metrics...")

if __name__ == "__main__":
    main()