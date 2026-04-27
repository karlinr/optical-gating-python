import logging

from hardware.system import SystemController
from app.config import Config

from logic.phase_estimator import PhaseManager

import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("Main")

def main():
    controller = SystemController()

    try:
        controller.connect_all()
        logger.info("All hardware components connected successfully.")

        controller.synchronise_camera()
        logger.info("Camera synchronisation complete.")

        controller.setup_cameras_for_experiment()
        controller.setup_timing_box_for_experiment()
        logger.info("Hardware setup for experiment complete. Ready to run.")

        phase_manager = PhaseManager()

        while True:
            frame, timestamp = controller.get_latest_bf_frame()
            # Convert time to ticks
            time_ticks = controller.timestamp_to_ticks(timestamp)

            # Get phase estimate
            print(phase_manager.update(frame))

            # Do prediction
            # Not implemented yet

            # Decide whether to fire
            # Not implemented yet

            # If firing, send command to timing box
            # Not implemented yet

            # If fired, get the latest frame from the fluorescence camera
            # Not implemented yet

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
    main()
