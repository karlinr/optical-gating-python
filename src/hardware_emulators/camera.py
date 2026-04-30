# src/hardware_emulators/camera.py
from loguru import logger
import threading
import time
import queue
import socket
import json
import numpy as np

class CameraEmulator:
    def __init__(self):
        self.serial_number = None
        self.is_running = False
        self.trigger_mode = False
        self.trigger_pin = None
        self.frame_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()

        self.x_coord = np.arange(512)
        self.y_coord = np.arange(512)
        self.X, self.Y = np.meshgrid(self.x_coord, self.y_coord)
        
        # Trigger listening
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 5005))
        self.sock.settimeout(0.1)

    def connect(self, serial_number):
        self.serial_number = serial_number
        logger.info(f"Emulator Camera SN {serial_number} connected.")

    def start_acquisition(self):
        self.is_running = True
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info(f"Emulator Camera SN {self.serial_number} acquisition started.")

    def _run_loop(self):
        while self.is_running and not self._stop_event.is_set():
            if self.trigger_mode:
                # Wait for trigger packet
                try:
                    data, _ = self.sock.recvfrom(1024)
                    states = json.loads(data.decode())
                    # Check if our configured trigger pin is active (High)
                    if states.get(str(self.trigger_pin), 0) == 1:
                        self._push_frame()
                except socket.timeout:
                    continue
            else:
                # Simulate frame rate delay
                time.sleep(1/80) 
                self._push_frame()

    def _push_frame(self):
        t = time.time()
        freq = 2  # 0.5 Hz = 2 second period
        
        # Calculate oscillating sizes (sigma)
        # base size of 25 pixels, oscillating by +/- 10
        # Blob 2 is pi/2 (90 degrees) out of phase with Blob 1
        sigma1 = 25 + 10 * np.sin(2 * np.pi * freq * t)
        sigma2 = 25 + 10 * np.sin(2 * np.pi * freq * t + np.pi/2)
        
        # Static positions for the "heart" chambers
        # Blob 1 (Left): (180, 256), Blob 2 (Right): (332, 256)
        blob1 = np.exp(-((self.X - 180)**2 + (self.Y - 256)**2) / (2 * sigma1**2))
        blob2 = np.exp(-((self.X - 332)**2 + (self.Y - 256)**2) / (2 * sigma2**2))
        
        # "Change it to a scale": Use the blobs to define the intensity scale
        # Base brightness of 100, scaled up to 255 by the blobs
        pattern = (blob1 + blob2) * 155
        
        # Add background noise (simulating sensor shot noise)
        noise = np.random.randint(50, 150, (512, 512), dtype=np.uint8)
        
        # Combine using addition and clip to avoid uint8 wrap-around
        frame = np.clip(pattern + noise, 0, 255).astype(np.uint8)

        if not self.frame_queue.full():
            self.frame_queue.put((frame, t))

    def get_latest_frame(self, timeout_ms=1000):
        return self.frame_queue.get(timeout=timeout_ms/1000.0)

    def stop_acquisition(self):
        self.is_running = False
        self._stop_event.set()
        logger.info(f"Emulator Camera SN {self.serial_number} stopped.")

    def set_mode_continuous(self, framerate=80):
        self.trigger_mode = False
        self.stop_acquisition()
        self.start_acquisition()

    def set_mode_hardware_trigger(self, source="XI_TRG_EDGE_RISING", cam_trigger_pin=0):
        self.trigger_mode = True
        self.trigger_pin = cam_trigger_pin
        self.stop_acquisition()
        self.start_acquisition()

    def close(self):
        self.is_running = False
        self._stop_event.set()
        logger.info(f"Emulator Camera SN {self.serial_number} closed.")