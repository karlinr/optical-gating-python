import pstats
import sys
from loguru import logger
import numpy as np
import zarr
from ome_zarr.writer import write_image
import tifffile as tf

from interfaces.system import SystemController
from app.config import Config
from app.state import AppState, ExperimentState

from logic.phase_estimator import PhaseManager
from logic.phase_predictor import BarrierPredictor

import cProfile
import matplotlib.pyplot as plt

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", enqueue=True)
logger.add("logs/main/experiment_{time}.log", rotation="10 MB", level="DEBUG", retention="10 days", enqueue=True)

# Key note to self: We primarily use the BF time as the current timestamp and only convert to ticks when sending commands to the timing box 

def main():
    app_state = AppState()
    controller = SystemController(app_state = app_state)

    storage_path = Config.ExperimentConfig.EXPERIMENT_DATA_PATH 

    firing = False   

    try:
        bf_test_frame, fl_test_frame = controller.connect_all()
        logger.info("All hardware components connected successfully.")

        app_state.set_state(ExperimentState.CALIBRATING)

        brightfield_chunk_size = (Config.ExperimentConfig.BRIGHTFIELD_CHUNK_SIZE, 1, 1, bf_test_frame.shape[0], bf_test_frame.shape[1])
        fl_chunk_size = (Config.ExperimentConfig.FLUORESCENCE_CHUNK_SIZE, 1, 1, fl_test_frame.shape[0], fl_test_frame.shape[1])

        logger.info(f"Brightfield camera frame shape: {bf_test_frame.shape}, chunk size: {brightfield_chunk_size}")
        logger.info(f"Fluorescence camera frame shape: {fl_test_frame.shape}, chunk size: {fl_chunk_size}")

        controller.synchronise_camera()
        controller.setup_cameras_for_experiment()
        controller.setup_timing_box_for_experiment()
        logger.success("Hardware setup for experiment complete. Ready to run.")

        app_state.set_state(ExperimentState.READY)

        phase_manager = PhaseManager(app_state = app_state)
        phase_predictor = BarrierPredictor()

        app_state.set_state(ExperimentState.RUNNING_EXPERIMENT)

        sad_phase_history = []
        mle_phase_history = []
        timestamp_history = []
        predicted_time_history = []
        trigger_time_history = []
        for i in range(5000):
            frame, timestamp = controller.get_latest_bf_frame()

            # Get phase estimate
            results = phase_manager.update(frame, timestamp = timestamp)
            sad_phase_history.append(results.get("SAD", {}).get("phase", None))
            mle_phase_history.append(results.get("MLE", {}).get("phase", None))
            timestamp_history.append(timestamp)
            logger.info(f"Status: {results["status"]}")

            # Do prediction
            if results["status"] == "READY":
                current_phase = results[Config.Gating.PHASE_SOURCE]["phase"]
                phase_predictor.update_phase(current_phase, timestamp)
                predicted_time = phase_predictor.predict_target_time(results["gating"]["target_phase"], results["gating"]["barrier_phase"])
                if predicted_time is not None:
                    predicted_time_history.append(predicted_time)

                    # Decide whether to fire the sequence based on predicted time and current time
                    time_to_fire = predicted_time - timestamp

                    if predicted_time is not None and timestamp is not None and 0.020 <= time_to_fire < 0.80 and not firing:
                        logger.info(f"Time to fire: {time_to_fire:.3f} seconds. Attempting to schedule fluorescence trigger at predicted time {predicted_time}...")
                        box_time, response = controller.trigger_fl_frame(predicted_time)
                        if response == 1:
                            logger.success(f"Successfully scheduled fluorescence trigger at predicted time {predicted_time}. Timing box response: {response}")
                            firing = True
                            fire_timestamp = predicted_time
                            trigger_time_history.append((timestamp, predicted_time))

            # If fired, get the latest frame from the fluorescence camera
            if firing:
                logger.info(f"Waiting for fluorescence frame to be captured after firing at predicted time {fire_timestamp}...")
                # Check if time is passed
                if timestamp > fire_timestamp + 0.1:  # Wait for 100 ms after predicted time to give the camera time to capture and transfer the frame
                    try:
                        fl_frame, fl_timestamp = controller.get_latest_fl_frame()
                        tf.imwrite(f"{storage_path}/fl_frame_{fl_timestamp}.tif", fl_frame)
                        firing = False
                        logger.success(f"Fluorescence frame captured at timestamp {fl_timestamp}")
                    except Exception as e:
                        firing = False
                        logger.error(f"Failed to capture fluorescence frame: {e}")

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(timestamp_history, sad_phase_history, label="Estimated Phase (SAD)")
        plt.plot(timestamp_history, mle_phase_history, label="Estimated Phase (MLE)")
        for i in range(len(trigger_time_history)):
            plt.axvline(x=trigger_time_history[i][0], color='r', linestyle='--', alpha=0.5, label="Fluorescence Trigger" if i == 0 else "")
        plt.xlabel("Time (s)")
        plt.ylabel("Phase (radians)")
        plt.title("Phase Estimation Over Time")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(timestamp_history[:len(predicted_time_history)], predicted_time_history, label="Predicted Time for Target Phase")
        plt.xlabel("Time (s)")
        plt.ylabel("Predicted Time (s)")
        plt.title("Phase Prediction Over Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

        unwrapped_phases = np.unwrap(np.array(sad_phase_history)[np.where(np.array(sad_phase_history) != None)])
        delta_phases = np.diff(unwrapped_phases)
        mle_unwrapped_phases = np.unwrap(np.array(mle_phase_history)[np.where(np.array(mle_phase_history) != None)])
        mle_phase_history

        plt.scatter(np.array(sad_phase_history)[np.where(np.array(sad_phase_history) != None)][1:], delta_phases, label="SAD")
        plt.scatter(np.array(mle_phase_history)[np.where(np.array(mle_phase_history) != None)][1:], np.diff(mle_unwrapped_phases), color="orange", label="MLE")
        plt.xlabel("Time (s)")
        plt.ylabel("Phase Difference (radians)")
        plt.title("Phase Estimation Comparison")
        plt.legend()
        plt.show()

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user. Shutting down.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        controller.timing_box.stop()
        controller.timing_box.close()
        controller.bf_cam.close()
        controller.fl_cam.close()
        logger.success("All hardware components shut down gracefully.")

if __name__ == "__main__":
    """profiler = cProfile.Profile()
    profiler.enable()"""

    main()

    """profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20)"""