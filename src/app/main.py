import pstats
import sys
from loguru import logger
import numpy as np

from interfaces.system import SystemController
from app.config import Config
from app.state import AppState, ExperimentState

from logic.phase_estimator import PhaseManager
from logic.phase_predictor import BarrierPredictor

import cProfile

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("logs/main/experiment_{time}.log", rotation="10 MB", level="DEBUG", retention="10 days")

# Key note to self: We primarily use the BF time as the current timestamp and only convert to ticks when sending commands to the timing box 

def main():
    app_state = AppState()
    controller = SystemController(app_state = app_state)

    try:
        controller.connect_all()
        logger.info("All hardware components connected successfully.")

        app_state.set_state(ExperimentState.CALIBRATING)
        controller.synchronise_camera()
        logger.info("Camera synchronisation complete.")

        controller.setup_cameras_for_experiment()
        controller.setup_timing_box_for_experiment()
        logger.info("Hardware setup for experiment complete. Ready to run.")

        app_state.set_state(ExperimentState.READY)

        phase_manager = PhaseManager(app_state = app_state)
        phase_predictor = BarrierPredictor()

        app_state.set_state(ExperimentState.RUNNING_EXPERIMENT)

        sad_phase_history = []
        mle_phase_history = []
        timestamp_history = []
        predicted_time_history = []
        for i in range(5000):
            frame, timestamp = controller.get_latest_bf_frame()

            # Get phase estimate
            results = phase_manager.update(frame, timestamp = timestamp)
            sad_phase_history.append(results.get("sad", {}).get("phase", None))
            mle_phase_history.append(results.get("mle", {}).get("phase", None))
            timestamp_history.append(timestamp)
            logger.info(f"Status: {results["status"]}")

            # Do prediction
            if results["status"] == "READY":
                current_phase = results["sad"]["phase"]
                phase_predictor.update_phase(current_phase, timestamp)
                predicted_time = phase_predictor.predict_target_time(results["sad"]["target_phase"], results["sad"]["barrier_phase"])
                if predicted_time is not None:
                    predicted_time_history.append(predicted_time)

            # Decide whether to fire
            # Not implemented yet

            # If firing, send command to timing box
            # Not implemented yet

            # If fired, get the latest frame from the fluorescence camera
            # Not implemented yet

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(timestamp_history, sad_phase_history, label="Estimated Phase (SAD)")
        plt.plot(timestamp_history, mle_phase_history, label="Estimated Phase (MLE)")
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

        plt.scatter(np.array(sad_phase_history)[np.where(np.array(sad_phase_history) != None)][1:], delta_phases)
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
        logger.info("All hardware components shut down gracefully.")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20)