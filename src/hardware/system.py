from hardware.timing_box import TimingBox
from hardware.camera import XimeaCamera
from app.config import Config

class SystemController:
    def __init__(self):
        self.timing_box = TimingBox()
        self.bf_cam = XimeaCamera()
        self.fl_cam = XimeaCamera()

    def connect_all(self):
        """
        Connects to the timing box and both cameras.
        """
        self.timing_box.connect()
        self.bf_cam.connect(serial_number = Config.BF.serial)
        self.fl_cam.connect(serial_number = Config.FL.serial)

    def synchronise_camera(self):
        """
        Here we want to set a common time that all future timestamps will be based upon.
        To do this we use the brightfield camera as the master clock and work out the conversion
        between the timing box's 24-bit clock and the camera's timestamp.
        Then in future if we want to trigger the fluorescence camera relative to the brightfield we can use the brightfield timestamp
        convert to timing box ticks and schedule a trigger at that time.
        """
        self.bf_cam.set_mode_hardware_trigger(cam_trigger_pin = Config.BF.trigger_pin)

        # Setup timing box trigger to trigger both cameras simultaneously
        self.timing_box.map_pin(Config.Hardware.Physical.BF, Config.Hardware.Logical.BF)  # Map physical pin to logical bit (camera trigger)

        # Now we want to trigger the brightfield camera twice, 1 second apart, and measure the timestamps and timing box ticks to work out the conversion factor
        # First setup our pianola memory to trigger the camera immediately and then 1 second later
        self.timing_box.add_step([Config.Hardware.Logical.BF], duration_ms = 100)  # Trigger brightfield camera
        # Turn off trigger
        self.timing_box.add_step([], duration_ms = 900)  # Wait for 0.9 seconds (total 1 second from first trigger)
        self.timing_box.add_step([Config.Hardware.Logical.BF], duration_ms = 100)  # Trigger brightfield camera again
        self.timing_box.finalize_sequence(repeat = False)

        # Now we run the sequence at a specific time so we know the exact timing box ticks for each trigger
        current_time = self.timing_box.get_current_time()
        fire_time, success = self.timing_box.fire_at(current_time + 100)  # Fire sequence after 100 ms to give us time to prepare
        # Now poll the brightfield camera for the two frames and record their timestamps and the corresponding timing box ticks
        bf_timestamps = []
        while len(bf_timestamps) < 2:
            frame, timestamp = self.bf_cam.get_latest_frame(timeout_ms = 2000)
            if timestamp is not None:
                bf_timestamps.append(timestamp)

        t1_box = fire_time
        t1_bf = bf_timestamps[0]
        t2_box = fire_time + 1000  # 1 second later in timing box ticks
        t2_bf = bf_timestamps[1]

        # Now we can calculate the conversion factor (gradient and intercept) between timing box ticks and camera timestamps
        # We have two points: (t1_box, t1_bf) and (t2_box, t2_bf)
        self.timestamp_to_ticks_gradient = (t2_box - t1_box) / (t2_bf - t1_bf)
        self.timestamp_to_ticks_intercept = t1_box - self.timestamp_to_ticks_gradient * t1_bf

    def setup_cameras_for_experiment(self):
        """
        Configures the cameras for the experiment.
        The brightfield camera is set to continuous (framerate) mode since we want it to run freely and provide timestamps.
        The fluorescence camera is set to hardware trigger mode since we only want it to capture frames when triggered by the timing box.
        """
        self.bf_cam.set_mode_continuous(framerate=Config.BF.framerate)
        self.fl_cam.set_mode_hardware_trigger(cam_trigger_pin = Config.FL.trigger_pin)

    def timestamp_to_ticks(self, timestamp):
        """
        Convert a camera timestamp to timing box ticks using the previously calculated conversion factor.
        All timestamps should be from the brightfield camera since that is our master clock.
        All triggers should be converted to timing box ticks and scheduled using fire_at() to ensure they are correctly synchronised to the camera's timeline.
        """
        if not hasattr(self, 'timestamp_to_ticks_gradient') or not hasattr(self, 'timestamp_to_ticks_intercept'):
            raise ValueError("Conversion factor not calculated. Please run synchronise_camera() first.")
        return int(self.timestamp_to_ticks_gradient * timestamp + self.timestamp_to_ticks_intercept)