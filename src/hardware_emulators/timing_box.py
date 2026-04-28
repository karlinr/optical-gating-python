"""
Emulates the timing box. Should be run in a separate process and will listen for commands on the specified serial port. It simulates the internal clock, hardware registers, and pin outputs of the timing box, allowing for testing and development without the physical hardware.
"""

import sys

import serial
import time
from loguru import logger
import threading
import json
import socket

# 1. Remove default handlers
logger.remove()

# 2. Add console handler (stdout)
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# 3. Add file handler (rotates every 10MB)
logger.add("logs/emulator/experiment_{time}.log", rotation="10 MB", level="DEBUG", retention="10 days")

from app.config import Config

class TimingBoxEmulator:
    TICK_SEC = 2.56e-6
    # ANSI Color Codes
    CLR_ACTIVE = ""#"\033[1;32m"  # Bold Green
    CLR_INACTIVE = ""#"\033[2;90m" # Dim Grey
    CLR_RESET = ""#"\033[0m"      # Reset formatting
    CLR_CMD = ""#"\033[1;34m"        # Bold Blue for commands
    CLR_DATA = ""#"\033[1;36m"       # Bold Cyan for data

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
        self.ser = serial.Serial(self.port, 115200, timeout=0.01) # Low timeout for responsiveness
        self.running_thread = None
        self.stop_signal = threading.Event()
        self.broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.broadcast_port = 5005
        self.reset_state()

    def reset_state(self):
        """Resets hardware registers and cancels pending fires."""
        self.start_perf = time.perf_counter()
        # Default: Phys 0-11 mapped to Log 0-7
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
        elapsed = time.perf_counter() - self.start_perf
        return int(elapsed / self.TICK_SEC) & 0xFFFFFF

    def sequence_executor(self):
        """Background thread simulating hardware pin output with numeric visualization."""
        current_step = 0
        self.stop_signal.clear()
        
        while self.is_running and not self.stop_signal.is_set():
            if current_step not in self.pianola_memory:
                break
            
            mask, duration_ticks = self.pianola_memory[current_step]
            self.logical_mask = mask
            
            # Construct Physical Pin Visualization
            phys_viz = ""
            for phys_idx in range(12):
                log_bit, invert = self.pin_mappings[phys_idx]
                is_active = ((self.logical_mask >> log_bit) & 1) ^ invert
                
                if is_active:
                    # Active: Bold Green with brackets
                    phys_viz += f"{self.CLR_ACTIVE}{phys_idx:02d}{self.CLR_RESET} "
                #else:
                #    # Inactive: Dim Grey
                #    phys_viz += f"{self.CLR_INACTIVE} {phys_idx:02d} {self.CLR_RESET} "
            
            # Construct Logical Bit Visualization
            log_viz = ""
            for i in range(8):
                if (self.logical_mask >> i) & 1:
                    log_viz += f"{self.CLR_ACTIVE}{i}{self.CLR_RESET}"
                #else:
                #    log_viz += f"{self.CLR_INACTIVE}{i}{self.CLR_RESET}"
            logger.info(f"STEP {current_step:02d} | LOGIC: {log_viz} | PHYS: {phys_viz}")

            pin_states = {i: ((self.logical_mask >> self.pin_mappings[i][0]) & 1) ^ self.pin_mappings[i][1] for i in range(12)}
            self.broadcast_socket.sendto(json.dumps(pin_states).encode(), ("127.0.0.1", self.broadcast_port))
            
            time.sleep(duration_ticks * self.TICK_SEC)
            
            if current_step >= self.final_step:
                if self.is_repeating:
                    current_step = self.repeat_from
                else:
                    self.is_running = False
            else:
                current_step += 1
        
        self.logical_mask = 0
        self.is_running = False
        logger.info(f"\n{self.CLR_RESET}[EMU] Sequence execution finished.")

    def handle_command(self, cmd):
        if cmd == self.CMDS["HARD_RESET"]: # HARD_RESET
            self.is_running = False
            self.stop_signal.set()
            self.reset_state()
            logger.info(f"CMD : {self.CLR_CMD}HARD_RESET{self.CLR_RESET}")

        elif cmd == self.CMDS["SET_PIANOLA"]: # SET_Pianola
            data = self.ser.read(5)
            if len(data) == 5:
                addr, mask, dur = data[0], data[1], int.from_bytes(data[2:5], 'big')
                self.pianola_memory[addr] = [mask, dur]
                logger.info(f"CMD : {self.CLR_CMD}SET_PIANOLA{self.CLR_RESET} | ADDR: {self.CLR_DATA}{addr}{self.CLR_RESET} MASK: {self.CLR_DATA}{mask:08b}{self.CLR_RESET} DUR: {self.CLR_DATA}{dur} ticks{self.CLR_RESET}")
            

        elif cmd == self.CMDS["SET_FINAL"]: # SET_PianolaFinalPos
            self.final_step = self.ser.read(1)[0]
            logger.info(f"CMD : {self.CLR_CMD}SET_FINAL{self.CLR_RESET} | FINAL STEP: {self.CLR_DATA}{self.final_step}{self.CLR_RESET}")

        elif cmd == self.CMDS["SET_REPEAT_FROM"]: # SET_PianolaRepeatFrom
            self.repeat_from = self.ser.read(1)[0]
            logger.info(f"CMD : {self.CLR_CMD}SET_REPEAT_FROM{self.CLR_RESET} | REPEAT FROM: {self.CLR_DATA}{self.repeat_from}{self.CLR_RESET}")

        elif cmd == self.CMDS["SET_REPEATING"]: # SET_PianolaRepeating
            self.is_repeating = bool(self.ser.read(1)[0])
            logger.info(f"CMD : {self.CLR_CMD}SET_REPEATING{self.CLR_RESET} | IS REPEATING: {self.CLR_DATA}{self.is_repeating}{self.CLR_RESET}")

        elif cmd == self.CMDS["RUN"]: # RUN_Pianola
            self.is_running = True
            self.ser.write(self.get_current_ticks().to_bytes(3, 'big'))
            self.running_thread = threading.Thread(target=self.sequence_executor, daemon=True)
            self.running_thread.start()
            logger.info(f"CMD : {self.CLR_CMD}RUN{self.CLR_RESET} | Sequence started at tick: {self.CLR_DATA}{self.get_current_ticks()}{self.CLR_RESET}")

        elif cmd == self.CMDS["FIRE_AT"]: # SET_PianolaFireTime
            target_bytes = self.ser.read(3)
            if len(target_bytes) == 3:
                self.scheduled_fire_time = int.from_bytes(target_bytes, 'big')
                current_ticks = self.get_current_ticks()
                
                # Check if target is in the future using 24-bit modular arithmetic
                # diff = (target - current) mod 2^24
                diff = (self.scheduled_fire_time - current_ticks) & 0xFFFFFF
                is_future = 1 if diff < 0x800000 else 0
                
                response = bytes([is_future]) + current_ticks.to_bytes(3, 'big')
                self.ser.write(response)
                logger.info(f"CMD : {self.CLR_CMD}FIRE_AT{self.CLR_RESET} | Target Tick: {self.CLR_DATA}{self.scheduled_fire_time}{self.CLR_RESET}")

        elif cmd == self.CMDS["STOP_RESET"]: # IRQ_StopAndReset
            self.is_running = False
            self.scheduled_fire_time = None
            self.stop_signal.set()
            logger.info(f"CMD : {self.CLR_CMD}STOP_RESET{self.CLR_RESET} | Sequence stopped and pending fires cleared.")

        elif cmd == self.CMDS["GET_TIME"]: # GET_CurrentPianolaTime
            self.ser.write(self.get_current_ticks().to_bytes(3, 'big'))
            logger.info(f"CMD : {self.CLR_CMD}GET_TIME{self.CLR_RESET} | Current Tick: {self.CLR_DATA}{self.get_current_ticks()}{self.CLR_RESET}")

        elif cmd == self.CMDS["MAP_PIN"]: # SET_PinSource
            data = self.ser.read(3)
            if len(data) == 3:
                self.pin_mappings[data[0]] = [data[1], data[2]]
                logger.info(f"CMD : {self.CLR_CMD}MAP_PIN{self.CLR_RESET} | PHYS: {self.CLR_DATA}{data[0]}{self.CLR_RESET} -> LOG: {self.CLR_DATA}{data[1]}{self.CLR_RESET} (Inverted: {bool(data[2])})")

        elif cmd == self.CMDS["GET_PIN_SOURCE"]: # GET_PinSource
            phys = self.ser.read(1)[0]
            self.ser.write(bytes(self.pin_mappings.get(phys, [0, 0])))
            logger.info(f"CMD : {self.CLR_CMD}GET_PIN_SOURCE{self.CLR_RESET} | PHYS: {self.CLR_DATA}{phys}{self.CLR_RESET} -> LOG: {self.CLR_DATA}{self.pin_mappings.get(phys, [0, 0])[0]}{self.CLR_RESET} (Inverted: {bool(self.pin_mappings.get(phys, [0, 0])[1])})")

    def check_scheduler(self):
        """Monitors the clock to trigger scheduled fire events."""
        if self.scheduled_fire_time is not None and not self.is_running:
            current = self.get_current_ticks()
            # Logic: If (current - target) mod 2^24 is small, current has passed target
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
                # Check serial buffer for new commands
                if self.ser.in_waiting > 0:
                    self.handle_command(self.ser.read(1)[0])
                
                # Check if it is time to fire a scheduled sequence
                self.check_scheduler()
                
                # Small sleep to prevent CPU spiking while maintaining precision
                time.sleep(0.0001)
        except KeyboardInterrupt:
            self.ser.close()

if __name__ == "__main__":
    # Ensure COM6 is your emulator port
    emu = TimingBoxEmulator(port= Config.TimingBox.EMULATOR_PORT)
    emu.run()