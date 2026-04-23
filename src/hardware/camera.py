import logging
import ximea.xiapi as xiapi

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("Camera")

class XimeaCamera:
    """
    Base class for interacting with Ximea cameras.
    """
    def __init(self):
        self.serial_number = None
        self.cam = xiapi.Camera()

    def connect(self, serial_number):
        self.serial_number = serial_number
        try:
            self.cam.open_device_by_SN(self.serial_number)
            logger.info(f"Camera with SN {self.serial_number} opened successfully.")
        except xiapi.XiError as e:
            logger.error(f"Failed to open camera with SN {self.serial_number}: {e}")
            raise 

    def get_latest_frame(self, timeout_ms=1000):
        try:
            self.cam.get_image(self.img_buffer, timeout_ms)

            timestamp = self.img_buffer.tsSec + self.img_buffer.tsUsec / 1e6
            frame_data = self.img_buffer.get_image_data_numpy()

            return frame_data, timestamp
        except xiapi.XiError as e:
            logger.error(f"Failed to get frame from camera SN {self.serial_number}: {e}")
            raise

    def start_acquisition(self):
        try:
            self.cam.start_acquisition()
            logger.info(f"Camera SN {self.serial_number} acquisition started.")
        except xiapi.XiError as e:
            logger.error(f"Failed to start acquisition on camera SN {self.serial_number}: {e}")
            raise

    def stop_acquisition(self):
        try:
            self.cam.stop_acquisition()
            logger.info(f"Camera SN {self.serial_number} acquisition stopped.")
        except xiapi.XiError as e:
            logger.error(f"Failed to stop acquisition on camera SN {self.serial_number}: {e}")
            raise

    def set_mode_continuous(self):
        """
        Sets the camera to continuous acquisition mode (no external trigger).
        Primarily used for the brightfield but can also be used for the fluorescence when setting up.
        """
        try:
            self.cam.set_trigger_source("XI_TRG_OFF")
            self.cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
            self.cam.set_framerate(60)
            logger.info(f"Camera SN {self.serial_number} set to continuous mode.")
        except xiapi.XiError as e:
            logger.error(f"Failed to set continuous mode on camera SN {self.serial_number}: {e}")
            raise

    def set_mode_hardware_trigger(self, source = "XI_TRG_EDGE_RISING", cam_trigger_pin = 0):
        """
        Sets the camera to hardware trigger mode.
        This is used for the fluorescence camera during gated acquisition and for timing synchronisation.
        """
        try:
            self.cam.set_trigger_source(source)
            self.cam.set_gpi_selector(f"XI_GPI_PORT{cam_trigger_pin}")
            self.cam.set_gpi_mode("XI_GPI_TRIGGER")
            logger.info(f"Camera SN {self.serial_number} set to hardware trigger mode (Source: {source}, GPI: {gpi}).")
        except xiapi.XiError as e:
            logger.error(f"Failed to set hardware trigger mode on camera SN {self.serial_number}: {e}")
            raise

    def close(self):
        if self.cam.is_device_opened():
            self.cam.close_device()
            logger.info(f"Camera with SN {self.serial_number} closed.")
        else:
            logger.warning("Camera is not open, cannot close.")

    