from loguru import logger
import ximea.xiapi as xiapi
import time

class XimeaCamera:
    """
    Base class for interacting with Ximea cameras.
    """
    def __init__(self):
        self.serial_number = None
        self.cam = xiapi.Camera()
        self.img_buffer = xiapi.Image()

        self.last_timestamp = None

    def connect(self, config):
        self.serial_number = config.serial
        try:
            self.cam.open_device_by_SN(self.serial_number)
            self.cam.set_debug_level(self.cam.get_debug_level_maximum())
            self.set_config(config)
            logger.info(f"Camera with SN {self.serial_number} opened successfully.")
        except xiapi.Xi_error as e:
            logger.error(f"Failed to open camera with SN {self.serial_number}: {e}")
            raise

    def set_config(self, config):
        logger.info(f"Setting camera ROI and dimensions for camera SN {self.serial_number}. ROI: {config.roi}, Downsample: {config.downsample}")
        if config.roi is not None:
            logger.info(f"Applying ROI settings for camera SN {self.serial_number}.")
            self.cam.set_width(config.roi[2])
            self.cam.set_height(config.roi[3])
            self.cam.set_offsetX(config.roi[0])
            self.cam.set_offsetY(config.roi[1])
        else:
            self.cam.set_width(self.cam.get_width_maximum())
            self.cam.set_height(self.cam.get_height_maximum())
            self.cam.set_offsetX(0)
            self.cam.set_offsetY(0)

        if config.sensor_taps is not None:
            logger.info(f"Setting sensor taps for camera SN {self.serial_number} to {config.sensor_taps}.")
            self.cam.set_sensor_taps(config.sensor_taps)

        logger.info(f"Setting camera SN {self.serial_number} exposure to {config.exposure_us} microseconds.")
        self.cam.set_exposure(config.exposure_us)

        time.sleep(0.1)  # Small delay to ensure settings are applied before starting acquisition

    def get_latest_frame(self, timeout_ms=1000):
        try:
            self.cam.get_image(self.img_buffer, timeout_ms)

            timestamp = self.img_buffer.tsSec + self.img_buffer.tsUSec / 1e6
            frame_data = self.img_buffer.get_image_data_numpy()

            if self.last_timestamp is not None:
                logger.info(f"Framerate: {1/(timestamp - self.last_timestamp)} FPS")
            self.last_timestamp = timestamp

            return frame_data, timestamp
        except xiapi.Xi_error as e:
            logger.error(f"Failed to get frame from camera SN {self.serial_number}: {e}")
            raise

    def start_acquisition(self):
        try:
            self.cam.start_acquisition()
            logger.info(f"Camera SN {self.serial_number} acquisition started.")
        except xiapi.Xi_error as e:
            logger.error(f"Failed to start acquisition on camera SN {self.serial_number}: {e}")
            raise

    def stop_acquisition(self):
        try:
            self.cam.stop_acquisition()
            logger.info(f"Camera SN {self.serial_number} acquisition stopped.")
        except xiapi.Xi_error as e:
            logger.error(f"Failed to stop acquisition on camera SN {self.serial_number}: {e}")
            raise

    def set_mode_continuous(self, framerate=60):
        """
        Sets the camera to continuous acquisition mode (no external trigger).
        Primarily used for the brightfield but can also be used for the fluorescence when setting up.
        """
        try:
            logger.info(f"Setting camera SN {self.serial_number} to continuous mode with framerate {framerate} FPS.")
            self.cam.stop_acquisition()
            logger.info(f"Camera SN {self.serial_number} acquisition stopped for mode switch.")
            self.cam.set_trigger_source("XI_TRG_OFF")
            logger.info(f"Camera SN {self.serial_number} trigger source set to XI_TRG_OFF (continuous mode).")
            self.cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
            logger.info(f"Camera SN {self.serial_number} acquisition timing mode set to XI_ACQ_TIMING_MODE_FRAME_RATE.")
            # Check if framerate is within allowed range
            if not (self.cam.get_framerate_minimum() <= framerate <= self.cam.get_framerate_maximum()):
                logger.warning(f"Requested framerate {framerate} is out of range for camera SN {self.serial_number}. Clamping to allowed range.")
                framerate = max(self.cam.get_framerate_minimum(), min(framerate, self.cam.get_framerate_maximum()))
                logger.info(f"Framerate for camera SN {self.serial_number} set to {framerate} FPS after clamping.")
            self.cam.set_framerate(framerate)
            logger.info(f"Camera SN {self.serial_number} framerate set to {framerate} FPS.")
            self.start_acquisition()
            logger.info(f"Camera SN {self.serial_number} set to continuous mode.")
        except xiapi.Xi_error as e:
            logger.error(f"Failed to set continuous mode on camera SN {self.serial_number}: {e}")
            raise

    def set_mode_hardware_trigger(self, source = "XI_TRG_EDGE_RISING", cam_trigger_pin = 0):
        """
        Sets the camera to hardware trigger mode.
        This is used for the fluorescence camera during gated acquisition and for timing synchronisation.
        """
        try:
            logger.info(f"Setting camera SN {self.serial_number} to hardware trigger mode with source {source} and trigger pin {cam_trigger_pin}.")
            self.cam.stop_acquisition()
            logger.info(f"Camera SN {self.serial_number} trigger source set to {source}.")
            self.cam.set_gpi_selector(f"XI_GPI_PORT{cam_trigger_pin}")
            logger.info(f"Camera SN {self.serial_number} GPI selector set to XI_GPI_PORT{cam_trigger_pin}.")
            self.cam.set_gpi_mode("XI_GPI_TRIGGER")
            logger.info(f"Camera SN {self.serial_number} GPI mode set to XI_GPI_TRIGGER.")
            logger.info(f"Camera SN {self.serial_number} acquisition stopped for mode switch.")
            self.cam.set_trigger_source(source)
            self.start_acquisition()
            logger.info(f"Camera SN {self.serial_number} set to hardware trigger mode (Source: {source}, GPI: {cam_trigger_pin}).")
        except xiapi.Xi_error as e:
            logger.error(f"Failed to set hardware trigger mode on camera SN {self.serial_number}: {e}")
            raise

    def close(self):
        self.cam.close_device()
        logger.info(f"Camera with SN {self.serial_number} closed.")

    