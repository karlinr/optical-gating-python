from loguru import logger
import threading
import time
import queue
import socket
import json
import numpy as np

class CameraEmulator:
    # Class-level shared socket resources to prevent Windows bind conflicts
    _shared_sock = None
    _listeners = []
    _lock = threading.Lock()
    _worker_thread = None

    def __init__(self):
        self.width = 32
        self.height = 32


        self.serial_number = None
        self.is_running = False
        self.trigger_mode = False
        self.trigger_pin = None
        self.framerate = 80  # Stored dynamically to calculate exact deadlines
        self.frame_queue = queue.Queue(maxsize=16)
        self._stop_event = threading.Event()
        self._trigger_event = threading.Event() # Internal signal for a hardware trigger

        self.t0 = time.perf_counter()

        # Pre-compute coordinates for frame generation
        self.x_coord = np.arange(self.width)
        self.y_coord = np.arange(self.height)
        self.X, self.Y = np.meshgrid(self.x_coord, self.y_coord)
        self.pixel_offsets = np.random.uniform(0, 2 * np.pi, (self.height, self.width)).astype(np.float32)
        
        # Initialize the shared listener
        self._ensure_shared_listener()

    @classmethod
    def _ensure_shared_listener(cls):
        """Starts a single background thread to listen for UDP trigger packets."""
        with cls._lock:
            if cls._shared_sock is None:
                cls._shared_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                cls._shared_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    cls._shared_sock.bind(("127.0.0.1", 5005))
                    cls._shared_sock.settimeout(0.5)
                    cls._worker_thread = threading.Thread(target=cls._socket_worker, daemon=True)
                    cls._worker_thread.start()
                    logger.success("Shared Emulator Listener started on port 5005.")
                except OSError as e:
                    logger.error(f"Failed to bind shared emulator socket: {e}")

    @classmethod
    def _socket_worker(cls):
        """Background worker that dispatches triggers to all camera instances."""
        while True:
            try:
                data, _ = cls._shared_sock.recvfrom(1024)
                states = json.loads(data.decode())
                with cls._lock:
                    for camera in cls._listeners:
                        camera._check_trigger(states)
            except (socket.timeout, OSError, json.JSONDecodeError):
                continue

    def _check_trigger(self, states):
        """Checks if this specific camera's GPIO pin was pulsed based on config.py mappings."""
        if self.trigger_mode and self.is_running:
            from app.config import Config
            
            # 1. Locate the camera configuration matching this instance's serial number
            matched_config = None
            if hasattr(Config.Cameras, 'BF') and Config.Cameras.BF.serial == self.serial_number:
                matched_config = Config.Cameras.BF
            elif hasattr(Config.Cameras, 'FL') and Config.Cameras.FL.serial == self.serial_number:
                matched_config = Config.Cameras.FL

            # 2. Extract and translate the pin mappings if a configuration was matched
            if matched_config is not None:
                for logical_bit in matched_config.box_pins:
                    try:
                        # Convert the logical bit value back to its string name (e.g., 1 -> 'FL_1')
                        pin_name = Config.TimingBox.Logical(logical_bit).name
                        # Look up the physical pin number assigned to that name (e.g., 'FL_1' -> 2)
                        physical_pin = int(Config.TimingBox.Physical[pin_name])
                        
                        # 3. Check if the matching physical pin is active in the broadcasted states
                        if states.get(str(physical_pin), 0) == 1:
                            self._trigger_event.set()
                            return  # Match found, exit early
                    except (ValueError, KeyError):
                        continue
            else:
                # Fallback to checking the raw unmapped trigger pin if no configuration matches
                if states.get(str(self.trigger_pin), 0) == 1:
                    self._trigger_event.set()

    def set_config(self, config):
        pass

    def connect(self, config_or_sn):
        """Connects the camera, handling either a raw string serial or a Config object."""
        if hasattr(config_or_sn, 'serial'):
            self.serial_number = config_or_sn.serial
        else:
            self.serial_number = config_or_sn

        with CameraEmulator._lock:
            if self not in CameraEmulator._listeners:
                CameraEmulator._listeners.append(self)
        logger.success(f"Emulator Camera SN {self.serial_number} connected.")

    def start_acquisition(self):
        # Clear out any frames remaining in the queue to emulate 
        # driver-level buffer initialization upon starting acquisition.
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.is_running = True
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info(f"Emulator Camera SN {self.serial_number} acquisition started.")

    def _run_loop(self):
        """High-precision, drift-corrected execution loop."""
        frame_interval = 1.0 / self.framerate
        next_frame_deadline = time.perf_counter()

        while self.is_running and not self._stop_event.is_set():
            if self.trigger_mode:
                # Trigger mode: Wait directly for hardware simulation events
                if self._trigger_event.wait(timeout=0.1):
                    self._push_frame()
                    self._trigger_event.clear()
            else:
                # Continuous mode: Enforce deterministic temporal intervals
                now = time.perf_counter()
                
                # Drop/skip catching up mechanisms if the host system hangs severely
                if now > next_frame_deadline + (2.0 * frame_interval):
                    next_frame_deadline = now

                # Calculate remaining time to wait until this iteration's target deadline
                sleep_duration = next_frame_deadline - now
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

                # Generate the physical data matrices
                self._push_frame()

                # Increment target deadline by a strict mathematical stride
                next_frame_deadline += frame_interval

    def _push_frame(self):
        """Generates a synthetic 16-bit 'heart' frame utilizing the full uint16 range."""
        t = time.perf_counter() - self.t0
        freq = 2
        
        global_phase = 2 * np.pi * freq * t
        
        # Vectorized 16-bit scaling: Midpoint baseline at 32,768 with a wave amplitude of 25,000
        pattern = 32768 + 25000 * np.cos(global_phase + self.pixel_offsets)
        
        # Generate 16-bit sensor background noise
        noise = np.random.randint(2000, 15000, (self.height, self.width), dtype=np.uint16)
        
        # Clamp strictly to maximum 16-bit integer boundaries
        frame = np.clip(pattern + noise, 0, 65535).astype(np.uint16)

        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass

        self.frame_queue.put((frame, t))

    def get_latest_frame(self, timeout_ms=1000):
        try:
            logger.debug(f"Camera SN {self.serial_number} - Waiting for frame. Current queue size: {self.frame_queue.qsize()}. Timeout: {timeout_ms} ms.")
            return self.frame_queue.get(timeout=timeout_ms/1000.0)
        except queue.Empty:
            return None, None

    def stop_acquisition(self):
        self.is_running = False
        self._stop_event.set()
        self._trigger_event.set() # Unblock the wait if triggered
        
        # Block the main thread until the background thread completely exits
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
            
        logger.info(f"Emulator Camera SN {self.serial_number} stopped.")

    def set_mode_continuous(self, framerate=80):
        self.trigger_mode = False
        self.framerate = framerate  # Explicitly bind the target parameter
        self.stop_acquisition()
        self.start_acquisition()

    def set_mode_hardware_trigger(self, source="XI_TRG_EDGE_RISING", cam_trigger_pin=0):
        self.trigger_mode = True
        self.trigger_pin = cam_trigger_pin
        self.stop_acquisition()
        self.start_acquisition()

    def close(self):
        self.stop_acquisition()
        with CameraEmulator._lock:
            if self in CameraEmulator._listeners:
                CameraEmulator._listeners.remove(self)
        logger.info(f"Emulator Camera SN {self.serial_number} closed.")