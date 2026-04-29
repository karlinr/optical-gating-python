from loguru import logger
import time

from interfaces.timing_box import TimingBox
from app.config import Config

if Config.EMULATE_CAMERA:
    from hardware_emulators.camera import CameraEmulator as XimeaCamera
else:
    from interfaces.camera import XimeaCamera

class SystemController:
    def __init__(self):
        self.timing_box = TimingBox(port = Config.TimingBox.PORT)
        self.bf_cam = XimeaCamera()
        self.fl_cam = XimeaCamera()

    def connect_all(self):
        """
        Connects to the timing box and both cameras.
        """
        self.timing_box.connect()
        self.bf_cam.connect(serial_number = Config.Cameras.BF.serial)
        self.fl_cam.connect(serial_number = Config.Cameras.FL.serial)

    def synchronise_camera(self):
        """
        Here we want to set a common time that all future timestamps will be based upon.
        To do this we use the brightfield camera as the master clock and work out the conversion
        between the timing box's 24-bit clock and the camera's timestamp.
        Then in future if we want to trigger the fluorescence camera relative to the brightfield we can use the brightfield timestamp
        convert to timing box ticks and schedule a trigger at that time.
        """
        pulse_ticks = TimingBox.to_24bit(0.1 / self.timing_box.TICK_SEC)  # 100 ms pulse duration in timing box ticks
        wait_ticks = TimingBox.to_24bit(0.9 / self.timing_box.TICK_SEC)   # 900 ms wait duration in timing box ticks (total 1 second cycle)
        total_time = pulse_ticks + wait_ticks

        self.bf_cam.set_mode_hardware_trigger(cam_trigger_pin = Config.Cameras.BF.trigger_pin)

        # Setup timing box trigger to trigger both cameras simultaneously
        self.timing_box.map_pin(Config.TimingBox.Physical.BF, Config.TimingBox.Logical.BF)  # Map physical pin to logical bit (camera trigger)

        # Now we want to trigger the brightfield camera twice, 1 second apart, and measure the timestamps and timing box ticks to work out the conversion factor
        # First setup our pianola memory to trigger the camera immediately and then 1 second later
        self.timing_box.add_step([Config.TimingBox.Logical.BF], duration_ticks=pulse_ticks)  # Trigger brightfield camera
        # Turn off trigger
        self.timing_box.add_step([], duration_ticks=wait_ticks)  # Wait for 0.9 seconds (total 1 second from first trigger)
        self.timing_box.add_step([Config.TimingBox.Logical.BF], duration_ticks=pulse_ticks)  # Trigger brightfield camera again
        self.timing_box.add_step([], duration_ticks=wait_ticks)  # Wait for 0.9 seconds (total 1 second from first trigger)
        self.timing_box.finalize_sequence(repeat = False)

        # Now we run the sequence at a specific time so we know the exact timing box ticks for each trigger
        current_time = self.timing_box.get_current_time()
        fire_time, success = self.timing_box.fire_at(current_time + TimingBox.to_24bit(0.1 / self.timing_box.TICK_SEC))  # Fire sequence after 100 ms to give us time to prepare
        # Now poll the brightfield camera for the two frames and record their timestamps and the corresponding timing box ticks
        bf_timestamps = []

        while TimingBox.is_future_tick(fire_time + total_time, self.timing_box.get_current_time()):
            time.sleep(0.1) # Small sleep to reduce CPU load

        while len(bf_timestamps) < 2:
            frame, timestamp = self.bf_cam.get_latest_frame(timeout_ms = 2000)
            if timestamp is not None:
                bf_timestamps.append(timestamp)

        t1_box = fire_time
        t1_bf = bf_timestamps[0]
        t2_box = fire_time + total_time # 1 second later in timing box ticks
        t2_bf = bf_timestamps[1]

        tick_delta = TimingBox.get_tick_diff(t2_box, t1_box)
        time_delta = t2_bf - t1_bf

        # Now we can calculate the conversion factor (gradient and intercept) between timing box ticks and camera timestamps
        # We have two points: (t1_box, t1_bf) and (t2_box, t2_bf)
        self.timestamp_to_ticks_gradient = tick_delta / time_delta
        self.timestamp_to_ticks_intercept = t1_box - self.timestamp_to_ticks_gradient * t1_bf

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
        We need to setup pin mappings
        and upload the pianola sequence that will be used to trigger the fl camera during the experiment.
        """
        self.timing_box.stop()

        # Use the pin mapping from the config
        self.timing_box.map_pin(Config.TimingBox.Physical.FL_1, Config.TimingBox.Logical.FL_1)

        # Upload the pianola sequence that will be used to trigger the fluorescence camera during the experiment
        # We should only have to do this once since we can use fire_at() to schedule it at the correct times during the experiment
        self.timing_box.add_step([Config.TimingBox.Logical.FL_1], duration_ticks=TimingBox.to_24bit(0.1 / self.timing_box.TICK_SEC))  # Trigger fluorescence camera
        self.timing_box.add_step([], duration_ticks=TimingBox.to_24bit(0.9 / self.timing_box.TICK_SEC))  # Wait for 0.9 seconds (total 1 second from first trigger)
        self.timing_box.finalize_sequence(repeat = False)


    def get_latest_bf_frame(self):
        """
        Retrieves the latest frame and timestamp from the brightfield camera.
        """
        return self.bf_cam.get_latest_frame()
    
    def trigger_fl_frame(self, timestamp: float):
        """Schedules a fluorescence trigger with safety checks for wrap-around."""
        target_tick = self.timestamp_to_ticks(timestamp)
        current_tick = self.timing_box.get_current_time()
        
        if not TimingBox.is_future_tick(target_tick, current_tick):
            logger.warning("Target trigger time has already passed in the 24-bit cycle.")
            return False, None
            
        return self.timing_box.fire_at(target_tick)

    def timestamp_to_ticks(self, timestamp):
        if not hasattr(self, 'timestamp_to_ticks_gradient'):
            raise ValueError("Run synchronise_camera() first.")
            
        raw_ticks = (self.timestamp_to_ticks_gradient * timestamp) + self.timestamp_to_ticks_intercept
        return TimingBox.to_24bit(raw_ticks)