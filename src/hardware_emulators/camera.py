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
                time.sleep(1/60) 
                self._push_frame()

    def _push_frame(self):
        # Create a dummy numpy frame
        frame = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        if not self.frame_queue.full():
            self.frame_queue.put((frame, time.time()))

    def get_latest_frame(self, timeout_ms=1000):
        try:
            return self.frame_queue.get(timeout=timeout_ms/1000.0)
        except queue.Empty:
            return np.zeros((512, 512)), time.time()

    def stop_acquisition(self):
        self.is_running = False
        self._stop_event.set()
        logger.info(f"Emulator Camera SN {self.serial_number} stopped.")

    def set_mode_continuous(self, framerate=60):
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