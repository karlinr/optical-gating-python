from loguru import logger
import time

from interfaces.timing_box import TimingBox
from app.config import Config

if Config.EMULATE_CAMERA:
    from emulators.camera import CameraEmulator as XimeaCamera
else:
    from interfaces.camera import XimeaCamera

class SystemController:
    def __init__(self):
        self.timing_box = TimingBox(port = Config.TimingBox.PORT)
        self.bf_cam = XimeaCamera()
        self.fl_cam = XimeaCamera()
        self.last_timestamp = 0

    def __enter__(self):
        """Allows SystemController to be used as a context manager."""
        logger.info("Entering SystemController hardware context.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Guarantees safe hardware teardown upon context exit."""
        logger.info("Exiting SystemController context. Commencing hardware teardown...")
        try:
            self.timing_box.stop()
            self.timing_box.close()
            self.bf_cam.close()
            self.fl_cam.close()
            logger.success("All hardware components shut down.")
        except Exception as e:
            logger.error(f"Error during hardware context cleanup: {e}")
        
        return False

    def connect_all(self):
        """
        Connects to the timing box and both cameras.
        """
        self.timing_box.connect()
        self.bf_cam.connect(Config.Cameras.BF)
        self.fl_cam.connect(Config.Cameras.FL)

        # Setup camera settings
        self.bf_cam.set_config(Config.Cameras.BF)
        self.fl_cam.set_config(Config.Cameras.FL)

        # Set each camera to trigger a single frame to verify the connection and get initial frame dimensions
        self.bf_cam.set_mode_hardware_trigger(cam_trigger_pin = Config.Cameras.BF.trigger_pin)
        self.fl_cam.set_mode_hardware_trigger(cam_trigger_pin = Config.Cameras.FL.trigger_pin)

        self._apply_camera_pin_mappings(Config.Cameras.BF)
        self._apply_camera_pin_mappings(Config.Cameras.FL)

        # Trigger a single frame on each camera to verify connection and get initial frame dimensions
        self.timing_box.add_step([Config.TimingBox.Logical.BF, Config.TimingBox.Logical.FL_1], duration_ticks=TimingBox.seconds_to_24bit_ticks(0.01))  # Trigger brightfield camera
        self.timing_box.add_step([], duration_ticks=TimingBox.seconds_to_24bit_ticks(0.05))  # Wait for 0.05 seconds (total 0.06 seconds from first trigger)
        self.timing_box.finalize_sequence(repeat = False)

        response = self.timing_box.run_now()

        if not response:
            logger.error("Failed to trigger cameras during connection test.")
            raise ConnectionError("Timing box failed to trigger cameras.")
        else:
            # Wait for camera frames and return them when ready
            bf_frame = None
            fl_frame = None
            while bf_frame is None or fl_frame is None:
                time.sleep(0.1) # Small sleep to reduce CPU load
                bf_frame, bf_timestamp = self.bf_cam.get_latest_frame(timeout_ms = 5000)
                fl_frame, fl_timestamp = self.fl_cam.get_latest_frame(timeout_ms = 5000)
                logger.info(f"Connection test: BF frame received: {bf_frame is not None}, FL frame received: {fl_frame is not None}")

        logger.info("Successfully connected to timing box and cameras.")
        return bf_frame, fl_frame

    def synchronise_camera(self):
        """
        Here we want to set a common time that all future timestamps will be based upon.
        To do this we use the brightfield camera as the master clock and work out the conversion
        between the timing box's 24-bit clock and the camera's timestamp.
        Then in future if we want to trigger the fluorescence camera relative to the brightfield we can use the brightfield timestamp
        convert to timing box ticks and schedule a trigger at that time.
        """
        self.timing_box.stop()

        pulse_ticks = TimingBox.seconds_to_24bit_ticks(0.01)
        wait_ticks_between_triggers = TimingBox.seconds_to_24bit_ticks(0.9)
        wait_time_to_trigger = TimingBox.seconds_to_24bit_ticks(0.5)
        
        sequence_gap_ticks = pulse_ticks + wait_ticks_between_triggers
        total_wait_time = wait_time_to_trigger + sequence_gap_ticks + pulse_ticks

        self.bf_cam.set_mode_hardware_trigger(cam_trigger_pin = Config.Cameras.BF.trigger_pin)

        # Setup timing box trigger to trigger both cameras simultaneously
        #self.timing_box.map_pin(Config.TimingBox.Physical.BF, Config.TimingBox.Logical.BF)  # Map physical pin to logical bit (camera trigger)
        self._apply_camera_pin_mappings(Config.Cameras.BF)

        # Now we want to trigger the brightfield camera twice and measure the timestamps and timing box ticks to work out the conversion factor
        # First setup our pianola memory to trigger the camera immediately
        self.timing_box.add_step([Config.TimingBox.Logical.BF], duration_ticks=pulse_ticks)
        # Turn off trigger
        self.timing_box.add_step([], duration_ticks=wait_ticks_between_triggers)
        self.timing_box.add_step([Config.TimingBox.Logical.BF], duration_ticks=pulse_ticks)
        self.timing_box.add_step([], duration_ticks=wait_ticks_between_triggers)
        self.timing_box.finalize_sequence(repeat = False)


        # Now we run the sequence at a specific time so we know the exact timing box ticks for each trigger
        current_time = self.timing_box.get_current_time()
        fire_at = (current_time + wait_time_to_trigger) & 0xFFFFFF
        logger.info(f"Current Timing Box tick: {current_time}, scheduling trigger at tick: {fire_at} (in {wait_time_to_trigger} ticks)")
        fire_time, success = self.timing_box.fire_at(fire_at)
        # Now poll the brightfield camera for the two frames and record their timestamps and the corresponding timing box ticks
        bf_timestamps = []

        while TimingBox.is_future_tick((fire_time + total_wait_time) & 0xFFFFFF, self.timing_box.get_current_time()):
            time.sleep(0.1)

        while len(bf_timestamps) < 2:
            frame, timestamp = self.bf_cam.get_latest_frame(timeout_ms = 2000)
            if timestamp is not None:
                bf_timestamps.append(timestamp)

        logger.info(f"Brightfield timestamps recorded: {bf_timestamps}")

        t1_box = fire_at
        t1_bf = bf_timestamps[0]
        t2_box = (fire_at + sequence_gap_ticks) & 0xFFFFFF  # 1 second later in timing box ticks
        t2_bf = bf_timestamps[1]

        tick_delta = TimingBox.get_tick_diff(t2_box, t1_box)
        time_delta = t2_bf - t1_bf

        # Now we can calculate the conversion factor (gradient and intercept) between timing box ticks and camera timestamps
        # We have two points: (t1_box, t1_bf) and (t2_box, t2_bf)
        self.timestamp_to_ticks_gradient = tick_delta / time_delta
        # Force gradient to expected value to avoid issues with small timing discrepancies during testing
        # NOTE: We may want to replace this. I will talk to Jonny about how we do this in SPIM GUI
        expected_gradient = 1/TimingBox.TICK_SEC
        self.timestamp_to_ticks_gradient = expected_gradient
        self.timestamp_to_ticks_intercept = t1_box - self.timestamp_to_ticks_gradient * t1_bf

        logger.info(f"Camera synchronisation complete. Timestamp to ticks conversion: ticks = {self.timestamp_to_ticks_gradient} * timestamp + {self.timestamp_to_ticks_intercept}")

    def setup_cameras_for_experiment(self):
        """
        Configures the cameras for the experiment.
        The brightfield camera is set to continuous (framerate) mode since we want it to run freely and provide timestamps.
        The fluorescence camera is set to hardware trigger mode since we only want it to capture frames when triggered by the timing box.
        """
        self.bf_cam.set_mode_continuous(framerate=Config.Cameras.BF.framerate)
        self.fl_cam.set_mode_hardware_trigger(cam_trigger_pin = Config.Cameras.FL.trigger_pin)

    def setup_timing_box_for_experiment(self):
        """
        Configures the timing box for the experiment.
        We need to setup pin mappings and upload the pianola sequence that will be used to trigger the fl camera during the experiment.
        """
        self.timing_box.stop()

        # Use the pin mapping from the config
        # This should trigger the camera and any associated lasers defined in the config
        self._apply_camera_pin_mappings(Config.Cameras.FL)

        # Upload the pianola sequence that will be used to trigger the fluorescence camera during the experiment
        # We should only have to do this once since we can use fire_at() to schedule it at the correct times during the experiment
        self.timing_box.add_step([Config.TimingBox.Logical.FL_1, Config.TimingBox.Logical.LAS_BLUE], duration_ticks=TimingBox.seconds_to_24bit_ticks(0.01))  # Trigger fluorescence camera
        self.timing_box.add_step([], duration_ticks=TimingBox.seconds_to_24bit_ticks(0.005))
        self.timing_box.finalize_sequence(repeat = False)

    def _apply_camera_pin_mappings(self, cam_config):
        for logical_bit in cam_config.box_pins:
            pin_name = Config.TimingBox.Logical(logical_bit).name
            physical_pin = Config.TimingBox.Physical[pin_name]
            self.timing_box.map_pin(physical_pin, logical_bit)
            logger.info(f"Mapped camera '{cam_config.label}' to Timing Box pin {physical_pin} (logical bit {logical_bit})")

    def get_latest_bf_frame(self):
        """
        Retrieves the latest frame and timestamp from the brightfield camera.
        """
        
        frame, timestamp = self.bf_cam.get_latest_frame()

        logger.debug(f"Framerate: {1 / (timestamp - self.last_timestamp)}, Timestamp: {timestamp}")

        framerate = 1 / (timestamp - self.last_timestamp) if self.last_timestamp else float('inf')

        self.last_timestamp = timestamp


        return frame, timestamp, framerate
    
    def get_latest_fl_frame(self):
        """
        Retrieves the latest frame and timestamp from the fluorescence camera.
        """
        try:
            frame, timestamp = self.fl_cam.get_latest_frame()
            return frame, timestamp
        except Exception as e:
            logger.error(f"Error getting frame from fluorescence camera: {e}")
            raise

    def trigger_fl_frame(self, timestamp: float):
        """Schedules a fluorescence trigger with safety checks for wrap-around."""
        target_tick = self.timestamp_to_ticks(timestamp)

        logger.info(f"Scheduling fluorescence trigger at timestamp {timestamp} (Timing Box tick: {target_tick})")

        return self.timing_box.fire_at(target_tick)

    def timestamp_to_ticks(self, timestamp):
        if not hasattr(self, 'timestamp_to_ticks_gradient'):
            raise ValueError("Run synchronise_camera() first.")
            
        raw_ticks = (self.timestamp_to_ticks_gradient * timestamp) + self.timestamp_to_ticks_intercept
        return TimingBox.to_24bit(raw_ticks)