from loguru import logger
import threading
import time
import queue
import socket
import json
import numpy as np
from app.config import Config

# ==============================================================================
# EMULATOR CONFIGURATION SETTINGS
# ==============================================================================
REPLAY_TIFF_FILE = True            
TIFF_FILE_PATH = r"../data/arrhyth.tif"  
#TIFF_FILE_PATH = r"../data/healthy_good_orientation.tif"
LOOP_TIFF = False                  # FORCED TRUE to prevent main loop flatlining

BASE_HEART_RATE_HZ = 2.0            
HEART_RATE_MODULATION_AMP = 0.4     
HEART_RATE_MODULATION_FREQ = 0.05    
# ==============================================================================

class CameraEmulator:
    _shared_sock = None
    _listeners = []
    _lock = threading.Lock()
    _worker_thread = None

    def __init__(self):
        self.width = 128
        self.height = 128
        self.serial_number = None
        self.is_running = False
        self.trigger_mode = False
        self.trigger_pin = None
        self.framerate = Config.Cameras.BF.framerate  
        self.frame_queue = queue.Queue(maxsize=16)
        self._stop_event = threading.Event()
        self._trigger_event = threading.Event() 

        self.t0 = time.perf_counter()
        self.accumulated_phase = 0.0
        self._last_phase_time = self.t0
        self.tiff_frames = None
        self.current_frame_idx = 0

        self.x_coord = np.arange(self.width)
        self.y_coord = np.arange(self.height)
        self.X, self.Y = np.meshgrid(self.x_coord, self.y_coord)
        self.pixel_offsets = np.random.uniform(0, 2 * np.pi, (self.height, self.width)).astype(np.float32)
        
        self._ensure_shared_listener()

    @classmethod
    def _ensure_shared_listener(cls):
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
        if self.trigger_mode and self.is_running:
            from app.config import Config
            matched_config = None
            if hasattr(Config.Cameras, 'BF') and Config.Cameras.BF.serial == self.serial_number:
                matched_config = Config.Cameras.BF
            elif hasattr(Config.Cameras, 'FL') and Config.Cameras.FL.serial == self.serial_number:
                matched_config = Config.Cameras.FL

            if matched_config is not None:
                for logical_bit in matched_config.box_pins:
                    try:
                        pin_name = Config.TimingBox.Logical(logical_bit).name
                        physical_pin = int(Config.TimingBox.Physical[pin_name])
                        if states.get(str(physical_pin), 0) == 1:
                            self._trigger_event.set()
                            return  
                    except (ValueError, KeyError):
                        continue
            else:
                if states.get(str(self.trigger_pin), 0) == 1:
                    self._trigger_event.set()

    def _load_tiff_file(self):
        if not REPLAY_TIFF_FILE:
            return
        logger.info(f"Camera SN {self.serial_number} - Loading TIFF from: {TIFF_FILE_PATH}")
        if hasattr(Config.Cameras, 'BF') and Config.Cameras.BF.serial == self.serial_number:
            matched_config = Config.Cameras.BF
        elif hasattr(Config.Cameras, 'FL') and Config.Cameras.FL.serial == self.serial_number:
            matched_config = Config.Cameras.FL
        if matched_config is not None:
            if matched_config.downsample == "XI_DWN_2x2":
                logger.info("Applying 2x2 downsampling to TIFF frames.")
                sampling_factor = 2
            elif matched_config.downsample == "XI_DWN_4x4":
                logger.info("Applying 4x4 downsampling to TIFF frames.")
                sampling_factor = 4
            else:
                sampling_factor = 1
        try:
            from PIL import Image
            img = Image.open(TIFF_FILE_PATH)
            frames = []
            try:
                while True:
                    frames.append(np.array(img))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
            self.tiff_frames = frames[:, ::sampling_factor, ::sampling_factor]
            self.height, self.width = self.tiff_frames[0].shape
            logger.success(f"Loaded {len(self.tiff_frames)} frames via PIL.")
        except Exception:
            try:
                import tifffile
                self.tiff_frames = tifffile.imread(TIFF_FILE_PATH)[::, ::sampling_factor, ::sampling_factor]
                self.height, self.width = self.tiff_frames[0].shape
                logger.success(f"Loaded {len(self.tiff_frames)} frames via tifffile.")
            except Exception as e:
                logger.error(f"IO Failure parsing TIFF: {e}")
                self.tiff_frames = None

    def set_config(self, config):
        pass

    def connect(self, config_or_sn):
        if hasattr(config_or_sn, 'serial'):
            self.serial_number = config_or_sn.serial
        else:
            self.serial_number = config_or_sn

        with CameraEmulator._lock:
            if self not in CameraEmulator._listeners:
                CameraEmulator._listeners.append(self)
        logger.success(f"Emulator Camera SN {self.serial_number} connected.")

    def start_acquisition(self):
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self._last_phase_time = time.perf_counter()
        self.current_frame_idx = 0
        if REPLAY_TIFF_FILE and self.tiff_frames is None:
            self._load_tiff_file()

        self.is_running = True
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        frame_interval = 1.0 / self.framerate
        next_frame_deadline = time.perf_counter()

        while self.is_running and not self._stop_event.is_set():
            if self.trigger_mode:
                if self._trigger_event.wait(timeout=0.1):
                    self._push_frame()
                    self._trigger_event.clear()
            else:
                now = time.perf_counter()
                if now > next_frame_deadline + (2.0 * frame_interval):
                    next_frame_deadline = now
                sleep_duration = next_frame_deadline - now
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                self._push_frame()
                next_frame_deadline += frame_interval

    def _push_frame(self):
        now = time.perf_counter()
        t = now - self.t0

        if REPLAY_TIFF_FILE and self.tiff_frames is not None:
            num_frames = len(self.tiff_frames)
            
            # --- CRITICAL LOOKUP FIX FOR TRIGGERED CHANNELS ---
            if self.trigger_mode:
                # Calculate the exact timeline-appropriate closest frame index
                self.current_frame_idx = int(round(t * self.framerate)) % num_frames
            else:
                # Continuous acquisition channel (Brightfield) auto-loops
                if self.current_frame_idx >= num_frames:
                    self.current_frame_idx = 0
            
            frame = self.tiff_frames[self.current_frame_idx].astype(np.uint16)
            
            if not self.trigger_mode:
                self.current_frame_idx = (self.current_frame_idx + 1) % num_frames
        else:
            # Synthetic Mode Fallback
            dt = now - self._last_phase_time
            self._last_phase_time = now
            #self.pixel_offsets += np.random.uniform(-0.005, 0.005, self.pixel_offsets.shape).astype(np.float32)
            self.pixel_offsets = np.mod(self.pixel_offsets, 2 * np.pi)
            freq = BASE_HEART_RATE_HZ + HEART_RATE_MODULATION_AMP * np.sin(2 * np.pi * HEART_RATE_MODULATION_FREQ * t)
            self.accumulated_phase += 2 * np.pi * freq * dt
            pattern = 32768 + 25000 * np.cos(self.accumulated_phase + self.pixel_offsets)
            noise = np.random.randint(2000, 15000, (self.height, self.width), dtype=np.uint16)
            frame = np.clip(pattern + noise, 0, 65535).astype(np.uint16)

        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass

        self.frame_queue.put((frame, t))

    def get_latest_frame(self, timeout_ms=1000):
        try:
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
        self._trigger_event.set()
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()

    def set_mode_continuous(self, framerate=80):
        self.trigger_mode = False
        self.framerate = framerate  
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