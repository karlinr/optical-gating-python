from loguru import logger
import threading
import time
import queue
import socket
import json
import numpy as np

# ==============================================================================
# EMULATOR CONFIGURATION SETTINGS
# ==============================================================================
# --- Mode Selector ---
REPLAY_TIFF_FILE = True            # Set to True to replay a real TIFF file; False for synthetic mode

# --- TIFF Replay Mode Settings ---
TIFF_FILE_PATH = r"C:\Users\Karlin\Documents\PhD\optical-gating-python\data\test_data.tif"  # Path to your multi-frame TIFF stack file
LOOP_TIFF = False

# --- Synthetic Heart Generation Settings ---
BASE_HEART_RATE_HZ = 2.0            # Baseline frequency of heart contraction (2.0 Hz = 120 BPM)
HEART_RATE_MODULATION_AMP = 0.4     # Amplitude of frequency variation (oscillates +/- 0.4 Hz)
HEART_RATE_MODULATION_FREQ = 0.05    # Frequency of modulation cycle in Hz (simulates sinus arrhythmia)
# ==============================================================================

class CameraEmulator:
    # Class-level shared socket resources to prevent Windows bind conflicts
    _shared_sock = None
    _listeners = []
    _lock = threading.Lock()
    _worker_thread = None

    def __init__(self):
        self.width = 10
        self.height = 10

        self.serial_number = None
        self.is_running = False
        self.trigger_mode = False
        self.trigger_pin = None
        self.framerate = 80  # Stored dynamically to calculate exact deadlines
        self.frame_queue = queue.Queue(maxsize=16)
        self._stop_event = threading.Event()
        self._trigger_event = threading.Event() # Internal signal for a hardware trigger

        self.t0 = time.perf_counter()
        
        # Phase tracking variables for continuous, jump-free frequency changes
        self.accumulated_phase = 0.0
        self._last_phase_time = self.t0

        # TIFF replay data structures
        self.tiff_frames = None
        self.current_frame_idx = 0

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

    def _load_tiff_file(self):
        """Loads a multi-frame TIFF file into memory using PIL or fallback to tifffile."""
        if not REPLAY_TIFF_FILE:
            return
        
        logger.info(f"Camera SN {self.serial_number} - Pre-loading TIFF stack from: {TIFF_FILE_PATH}")
        try:
            # Try loading via PIL (Pillow) first as it is broadly accessible standard
            from PIL import Image
            img = Image.open(TIFF_FILE_PATH)
            frames = []
            try:
                while True:
                    frames.append(np.array(img))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
            
            self.tiff_frames = frames
            if len(self.tiff_frames) == 0:
                raise ValueError("TIFF stack contains zero frames.")
            
            # Re-scale matrix properties to conform with physical file footprint
            self.height, self.width = self.tiff_frames[0].shape
            logger.success(f"Successfully loaded {len(self.tiff_frames)} frames via PIL. Resolution: {self.width}x{self.height}")
            
        except Exception as e_pil:
            logger.warning(f"PIL parser failed ({e_pil}). Attempting fallback execution via 'tifffile'...")
            try:
                import tifffile
                self.tiff_frames = tifffile.imread(TIFF_FILE_PATH)
                if len(self.tiff_frames) == 0:
                    raise ValueError("TIFF stack contains zero frames.")
                self.height, self.width = self.tiff_frames[0].shape
                logger.success(f"Successfully loaded {len(self.tiff_frames)} frames via tifffile. Resolution: {self.width}x{self.height}")
            except Exception as e_tiff:
                logger.error(f"Catastrophic File IO Failure: Unable to parse TIFF image stack via PIL or tifffile: {e_tiff}")
                self.tiff_frames = None

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

        # Reset time references to guarantee smooth sequence initialization
        self._last_phase_time = time.perf_counter()
        self.current_frame_idx = 0
        
        # Load the target stack image data into memory if enabled
        if REPLAY_TIFF_FILE and self.tiff_frames is None:
            self._load_tiff_file()

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
        """Generates a synthetic 16-bit frame or streams the next recorded frame from a TIFF stack."""
        now = time.perf_counter()
        t = now - self.t0

        if REPLAY_TIFF_FILE and self.tiff_frames is not None:
            # --- MODE A: TIFF HARDWARE FILE REPLAY ---
            num_frames = len(self.tiff_frames)
            if self.current_frame_idx >= num_frames:
                if LOOP_TIFF:
                    self.current_frame_idx = 0
                else:
                    logger.warning(f"Camera SN {self.serial_number} - End of TIFF data reached. Frozen on final frame.")
                    self.current_frame_idx = num_frames - 1
            
            frame = self.tiff_frames[self.current_frame_idx].astype(np.uint16)
            
            # Advance structural marker ready for next interval interval or hardware sync edge pulse
            if LOOP_TIFF or self.current_frame_idx < num_frames - 1:
                self.current_frame_idx = (self.current_frame_idx + 1) % num_frames
        else:
            # --- MODE B: SYNTHETIC MODEL WITH CONTINUOUS HEARTRATE DRIFT ---
            dt = now - self._last_phase_time
            self._last_phase_time = now

            # Add a small amount of random jitter to the model pixel_offsets to simulate the model changing over time
            self.pixel_offsets += np.random.uniform(-0.005, 0.005, self.pixel_offsets.shape).astype(np.float32)
            self.pixel_offsets = np.mod(self.pixel_offsets, 2 * np.pi)

            # Compute immediate localized target cardiac cycle frequency
            freq = BASE_HEART_RATE_HZ + HEART_RATE_MODULATION_AMP * np.sin(2 * np.pi * HEART_RATE_MODULATION_FREQ * t)
            
            # Smoothly integrate state step changes via slice deltas to preserve continuous signal phase
            self.accumulated_phase += 2 * np.pi * freq * dt
            
            # Vectorized 16-bit scaling: Midpoint baseline at 32,768 with a wave amplitude of 25,000
            pattern = 32768 + 25000 * np.cos(self.accumulated_phase + self.pixel_offsets)
            
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
            frame, t = self.frame_queue.get(timeout=timeout_ms / 1000.0)
            metadata = {
                "true_phase": self.accumulated_phase % (2 * np.pi),
                "frame_idx": self.current_frame_idx,
                "is_synthetic": self.tiff_frames is None
            }
            return frame, t, metadata
        except queue.Empty:
            return None, None, {}

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