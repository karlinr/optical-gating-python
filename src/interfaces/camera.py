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

    def connect(self, config):
        self.serial_number = config.serial
        try:
            self.cam.open_device_by_SN(self.serial_number)
            self.cam.set_debug_level(self.cam.get_debug_level_maximum())
            time.sleep(0.1)
            self.set_config(config)
            logger.info(f"Camera with SN {self.serial_number} opened successfully.")
        except xiapi.Xi_error as e:
            logger.error(f"Failed to open camera with SN {self.serial_number}: {e}")
            raise

    def set_config(self, config):
        try:
            self.cam.stop_acquisition()
        except xiapi.Xi_error:
            pass

        self.cam.set_offsetX(0)
        self.cam.set_offsetY(0)

        if config.sensor_taps is not None:
            try:
                self.cam.set_sensor_taps(config.sensor_taps)
            except Exception:
                logger.warning(f"Sensor taps unsupported on SN {self.serial_number}.")

        if config.downsample is not None:
            try:
                self.cam.set_downsampling(config.downsample)
            except Exception:
                logger.warning(f"Downsampling unsupported on SN {self.serial_number}.")

        factor = self._downsample_factor()
        logger.info(f"SN {self.serial_number} downsample factor: {factor}, "
                    f"new max: ({self.cam.get_width_maximum()}, {self.cam.get_height_maximum()})")

        if config.roi is not None:
            x, y, w, h = (v // factor for v in config.roi)

            w = min(self._align_down(w, self.cam.get_width_increment()),
                    self.cam.get_width_maximum())
            h = min(self._align_down(h, self.cam.get_height_increment()),
                    self.cam.get_height_maximum())
            self.cam.set_width(w)
            self.cam.set_height(h)

            x = min(self._align_down(x, self.cam.get_offsetX_increment()),
                    self.cam.get_offsetX_maximum())
            y = min(self._align_down(y, self.cam.get_offsetY_increment()),
                    self.cam.get_offsetY_maximum())
            self.cam.set_offsetX(x)
            self.cam.set_offsetY(y)

            logger.info(f"Applied ROI for SN {self.serial_number}: "
                        f"full {config.roi} -> offset=({x},{y}) size=({w},{h})")
        else:
            self.cam.set_width(self.cam.get_width_maximum())
            self.cam.set_height(self.cam.get_height_maximum())

        self.cam.set_exposure(config.exposure_us)
        time.sleep(0.1)

    def get_latest_frame(self, timeout_ms=1000):
        try:
            self.cam.get_image(self.img_buffer, timeout_ms)

            timestamp = self.img_buffer.tsSec + self.img_buffer.tsUSec / 1e6
            frame_data = self.img_buffer.get_image_data_numpy()
            metadata = {
                "is_synthetic": False
            }

            return frame_data, timestamp, metadata
        except xiapi.Xi_error as e:
            logger.error(f"Failed to get frame from camera SN {self.serial_number}: {e}")
            raise

    def start_acquisition(self):
        try:
            self.cam.start_acquisition()
            logger.info(f"Camera SN {self.serial_number} acquisition started.")
        except xiapi.Xi_error as e:
            logger.error(f"Failed to start acquisition on camera SN {self.serial_number}: {e}")

    def stop_acquisition(self):
        try:
            self.cam.stop_acquisition()
            logger.info(f"Camera SN {self.serial_number} acquisition stopped.")
        except xiapi.Xi_error as e:
            logger.error(f"Failed to stop acquisition on camera SN {self.serial_number}: {e}")

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
        try:
            self.stop_acquisition()
        except:
            pass
        self.cam.close_device()
        logger.info(f"Camera with SN {self.serial_number} closed.")

    def _downsample_factor(self):
        """Return the current downsampling factor as an int (1, 2, 4...)."""
        val = self.cam.get_downsampling()
        # Ximea returns either an int-like value or a string such as 'XI_DWN_2x2'
        s = str(val)
        if "x" in s.lower():
            # e.g. "XI_DWN_2x2" -> "2"
            digits = "".join(ch for ch in s.split("_")[-1].split("x")[0] if ch.isdigit())
            return int(digits) if digits else 1
        try:
            return int(val)
        except (TypeError, ValueError):
            return 1
    
    def _align_down(self, value, increment):
        if increment and increment > 0:
            return (value // increment) * increment
        return value