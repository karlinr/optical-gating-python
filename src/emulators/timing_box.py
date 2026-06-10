"""
Emulators the timing box with thread-safe operations, fixed rejection states,
and an adaptive real-time scheduler to maintain microsecond accuracy without high CPU load.
"""

import sys
import serial
import time
from loguru import logger
import threading
import json
import socket

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
#logger.add("logs/emulator/experiment_{time}.log", rotation="10 MB", level="DEBUG", retention="10 days")

from app.config import Config

class TimingBoxEmulator:
    TICK_SEC = 2.56e-6
    CLR_ACTIVE = ""
    CLR_RESET = ""
    CLR_CMD = ""
    CLR_DATA = ""

    CMDS = {
        "SET_PIANOLA": 0x01,
        "SET_FINAL": 0x02,
        "SET_REPEAT_FROM": 0x03,
        "SET_REPEATING": 0x04,
        "RUN": 0x05,
        "FIRE_AT": 0x06,
        "STOP_RESET": 0x07,
        "GET_TIME": 0x08,
        "MAP_PIN": 0x09,
        "GET_PIN_SOURCE": 0x0A,
        "DUMP_LOG": 0xFE,
        "HARD_RESET": 0xFF
    }

    def __init__(self, port):
        self.port = port
        self.ser = serial.Serial(self.port, 115200, timeout=0.01)
        self.running_thread = None
        self.stop_signal = threading.Event()
        self.broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.broadcast_port = 5005
        
        # --- STRUCTURAL FIX: UPGRADE TO REENTRANT LOCK ---
        self.lock = threading.RLock()
        self.reset_state()

    def reset_state(self):
        """Resets hardware registers and cancels pending fires."""
        with self.lock:
            self.start_perf = time.perf_counter()
            self.pin_mappings = {i: [i if i < 8 else 0, 0] for i in range(12)}
            self.pianola_memory = {} 
            self.final_step = 0
            self.repeat_from = 0
            self.is_repeating = False
            self.is_running = False
            self.logical_mask = 0 
            self.scheduled_fire_time = None

    def get_current_ticks(self):
        """Calculates current 24-bit clock ticks."""
        with self.lock:
            elapsed = time.perf_counter() - self.start_perf
        return int(elapsed / self.TICK_SEC) & 0xFFFFFF

    def sequence_executor(self):
        """Background thread simulating hardware pin output with numeric visualization."""
        current_step = 0
        self.stop_signal.clear()
        
        while True:
            with self.lock:
                if not self.is_running or self.stop_signal.is_set() or current_step not in self.pianola_memory:
                    break
                mask, duration_ticks = self.pianola_memory[current_step]
                self.logical_mask = mask
                pin_mappings_copy = self.pin_mappings.copy()
            
            # Numeric output visualization formatting
            phys_viz = ""
            for phys_idx in range(12):
                log_bit, invert = pin_mappings_copy[phys_idx]
                is_active = ((mask >> log_bit) & 1) ^ invert
                if is_active:
                    phys_viz += f"{phys_idx:02d} "
            
            log_viz = "".join([str(i) for i in range(8) if (mask >> i) & 1])
            logger.info(f"STEP {current_step:02d} | LOGIC: {log_viz} | PHYS: {phys_viz}")

            pin_states = {i: ((mask >> pin_mappings_copy[i][0]) & 1) ^ pin_mappings_copy[i][1] for i in range(12)}
            try:
                self.broadcast_socket.sendto(json.dumps(pin_states).encode(), ("127.0.0.1", self.broadcast_port))
            except OSError:
                pass
            
            # Target tick constraint validation
            target_tick = (self.get_current_ticks() + duration_ticks) & 0xFFFFFF
            while True:
                with self.lock:
                    if not self.is_running or self.stop_signal.is_set():
                        break
                now = self.get_current_ticks()
                if ((now - target_tick) & 0xFFFFFF) < 0x800000:
                    break
                time.sleep(0.0005)
            
            with self.lock:
                if current_step >= self.final_step:
                    if self.is_repeating:
                        current_step = self.repeat_from
                    else:
                        self.is_running = False
                        break
                else:
                    current_step += 1
        
        with self.lock:
            self.logical_mask = 0
            self.is_running = False
        logger.info("\n[EMU] Sequence execution finished.")

    def handle_command(self, cmd):
        if cmd == self.CMDS["HARD_RESET"]:
            with self.lock:
                self.is_running = False
            self.stop_signal.set()
            self.reset_state()
            logger.info("CMD : HARD_RESET")

        elif cmd == self.CMDS["SET_PIANOLA"]:
            data = self.ser.read(5)
            if len(data) == 5:
                addr, mask, dur = data[0], data[1], int.from_bytes(data[2:5], 'big')
                with self.lock:
                    self.pianola_memory[addr] = [mask, dur]
                logger.info(f"CMD : SET_PIANOLA | ADDR: {addr} MASK: {mask:08b} DUR: {dur} ticks")

        # --- STRUCTURAL FIXES: LENGTH-GUARDED SHORT READS ---
        elif cmd == self.CMDS["SET_FINAL"]:
            data = self.ser.read(1)
            if len(data) == 1:
                with self.lock:
                    self.final_step = data[0]
                logger.info(f"CMD : SET_FINAL | FINAL STEP: {self.final_step}")

        elif cmd == self.CMDS["SET_REPEAT_FROM"]:
            data = self.ser.read(1)
            if len(data) == 1:
                with self.lock:
                    self.repeat_from = data[0]
                logger.info(f"CMD : SET_REPEAT_FROM | REPEAT FROM: {self.repeat_from}")

        elif cmd == self.CMDS["SET_REPEATING"]:
            data = self.ser.read(1)
            if len(data) == 1:
                with self.lock:
                    self.is_repeating = bool(data[0])
                logger.info(f"CMD : SET_REPEATING | IS REPEATING: {self.is_repeating}")

        elif cmd == self.CMDS["RUN"]:
            with self.lock:
                self.is_running = True
            current_ticks = self.get_current_ticks()
            self.ser.write(current_ticks.to_bytes(3, 'big'))
            self.running_thread = threading.Thread(target=self.sequence_executor, daemon=True)
            self.running_thread.start()
            logger.info(f"CMD : RUN | Sequence started at tick: {current_ticks}")

        elif cmd == self.CMDS["FIRE_AT"]:
            target_bytes = self.ser.read(3)
            if len(target_bytes) == 3:
                requested_fire_time = int.from_bytes(target_bytes, 'big')
                current_ticks = self.get_current_ticks()
                
                diff = (requested_fire_time - current_ticks) & 0xFFFFFF
                is_future = 1 if diff < 0x800000 else 0

                with self.lock:
                    if is_future:
                        self.scheduled_fire_time = requested_fire_time
                    else:
                        self.scheduled_fire_time = None
                        logger.warning(f"FIRE_AT target {requested_fire_time} rejected (In the past relative to tick {current_ticks}).")
                
                response = bytes([is_future]) + current_ticks.to_bytes(3, 'big')
                self.ser.write(response)
                logger.info(f"CMD : FIRE_AT | Target Tick: {requested_fire_time} | Status: {'Accepted' if is_future else 'Rejected'}")

        elif cmd == self.CMDS["STOP_RESET"]:
            with self.lock:
                self.is_running = False
                self.scheduled_fire_time = None
            self.stop_signal.set()
            logger.info("CMD : STOP_RESET | Sequence stopped and pending fires cleared.")

        elif cmd == self.CMDS["GET_TIME"]:
            current_ticks = self.get_current_ticks()
            self.ser.write(current_ticks.to_bytes(3, 'big'))
            logger.info(f"CMD : GET_TIME | Current Tick: {current_ticks}")

        elif cmd == self.CMDS["MAP_PIN"]:
            data = self.ser.read(3)
            if len(data) == 3:
                with self.lock:
                    self.pin_mappings[data[0]] = [data[1], data[2]]
                logger.info(f"CMD : MAP_PIN | PHYS: {data[0]} -> LOG: {data[1]} (Inverted: {bool(data[2])})")

        elif cmd == self.CMDS["GET_PIN_SOURCE"]:
            data = self.ser.read(1)
            if len(data) == 1:
                phys = data[0]
                with self.lock:
                    mapping = self.pin_mappings.get(phys, [0, 0])
                self.ser.write(bytes(mapping))
                logger.info(f"CMD : GET_PIN_SOURCE | PHYS: {phys} -> LOG: {mapping[0]} (Inverted: {bool(mapping[1])})")

    def check_scheduler(self):
        """Monitors the clock to trigger scheduled fire events."""
        with self.lock:
            if self.scheduled_fire_time is None or self.is_running:
                return

            current = self.get_current_ticks()
            if (current - self.scheduled_fire_time) & 0xFFFFFF < 0x800000:
                logger.info(f"Scheduled fire triggered at tick {current}")
                self.scheduled_fire_time = None
                self.is_running = True
                self.running_thread = threading.Thread(target=self.sequence_executor, daemon=True)
                self.running_thread.start()

    def run(self):
        logger.info(f"Emulator active on {self.port}.")
        try:
            while True:
                if self.ser.in_waiting > 0:
                    # Guard the initial instruction fetch
                    cmd_byte = self.ser.read(1)
                    if len(cmd_byte) == 1:
                        self.handle_command(cmd_byte[0])
                
                self.check_scheduler()
                
                with self.lock:
                    fire_time = self.scheduled_fire_time
                    running = self.is_running
                
                if fire_time is not None and not running:
                    ticks_until_fire = (fire_time - self.get_current_ticks()) & 0xFFFFFF
                    if ticks_until_fire > 1500:
                        time.sleep(0.001)
                    else:
                        pass
                else:
                    time.sleep(0.002)
        except KeyboardInterrupt:
            self.ser.close()

if __name__ == "__main__":
    emu = TimingBoxEmulator(port=Config.TimingBox.EMULATOR_PORT)
    emu.run()