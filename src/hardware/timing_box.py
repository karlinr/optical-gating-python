import serial
import logging
import time

from app.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("TimingBox")

class TimingBox:
    TICK_SEC = 2.56e-6
    MAX_TICKS = 0xFFFFFF     # 16,777,215
    HALF_RANGE = 0x800000    # 8,388,608

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
        self.ser = None
        self.step_index = 0

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, 115200, timeout=1.0)
            logger.info(f"Serial connection established on {self.port}")
            for i in range(6):
                self.hard_reset()
                time.sleep(0.05)
                logger.info(f"Sent HARD_RESET command ({i+1}/6)")
            logger.info(f"Connected to Timing Box on {self.port}")
        except serial.SerialException as e:
            logger.error(f"Connection failed: {e}")

    def _send_command(self, cmd_name, data=None):
        # Check we are connected before sending
        if not (self.ser and self.ser.is_open):
            logger.error("Serial connection is not open")
            return

        # Verify command exists
        cmd_byte = self.CMDS.get(cmd_name)
        if cmd_byte is None:
            logger.error(f"Invalid command: {cmd_name}")
            return
        
        logger.info(f"Sending command: {cmd_name} with data: {data}")

        # Construct payload: [Command Byte, Data...] and send
        payload = bytearray([cmd_byte, *(data or [])])
        self.ser.write(payload)
        self.ser.flush()
        logger.info(f"Sent {cmd_name}: {payload.hex()}")

    def map_pin(self, physical_pin, logical_pin, invert=False):
        """
        Maps a physical pin to a specific bit in the pianola mask.
        """
        if not (0 <= physical_pin < 12):
            logger.error(f"Physical pin {physical_pin} is out of hardware range (0-11)")
            return
        
        if not (0 <= logical_pin < 8):
            logger.error(f"Logical pin {logical_pin} must be in range 0-7")
            return

        # Payload: [pin index, bit index, inversion flag]
        payload = [int(physical_pin), int(logical_pin), int(invert)]
        self._send_command("MAP_PIN", payload)

    def get_pin_mapping(self, pin):
        """
        Queries the hardware for the mapping of a specific physical pin.
        Returns a tuple: (source_bit, is_inverted)
        """
        if not (0 <= pin < 12):
            logger.error(f"Pin {pin} is out of range (0-11)")
            return None

        # Send command 0x0A and the physical pin index
        self._send_command("GET_PIN_SOURCE", [pin])

        # The C++ code expects a 2-byte response:
        # byte 0: the source bit (0-7)
        # byte 1: the inversion flag (0 or 1)
        response = self.ser.read(2)

        if len(response) == 2:
            source_bit = response[0]
            is_inverted = bool(response[1])
            return source_bit, is_inverted

        logger.error(f"Failed to receive pin mapping response for pin {pin}")
        return None

    def add_step(self, logical_pins, duration_ticks):
        """
        Adds a step to the sequence. 
        Duration is converted to 2.56us ticks and sent as 3 bytes (Big Endian).
        """
        mask = sum(1 << int(p) for p in set(logical_pins))
        ticks = max(1, int(duration_ticks))

        # Payload: [Address, Mask, Dur_High, Dur_Mid, Dur_Low]
        duration_bytes = list(int(ticks & 0xFFFFFF).to_bytes(3, 'big'))
        payload = [self.step_index, mask & 0xFF] + duration_bytes
        
        self._send_command("SET_PIANOLA", payload)
        self.step_index += 1

    def finalize_sequence(self, repeat=False, loop_from=0):
        """Sets the end of the sequence and optional looping."""
        last_step = max(0, self.step_index - 1)
        self._send_command("SET_FINAL", [last_step])
        self._send_command("SET_REPEAT_FROM", [loop_from])
        self._send_command("SET_REPEATING", [1 if repeat else 0])

    def run_now(self):
        """Executes the loaded sequence once immediately."""
        self._send_command("RUN")
        # Returns 3-byte clock time
        response = self.ser.read(3)
        return int.from_bytes(response, 'big') if len(response) == 3 else None

    def fire_at(self, tick_timestamp):
        """Schedules the sequence to run at a specific absolute tick count."""
        # Uses 3-byte time
        data = list(int(tick_timestamp & 0xFFFFFF).to_bytes(3, 'big'))
        self._send_command("FIRE_AT", data)
        # Returns 1-byte flag and 3-byte clock
        response = self.ser.read(4)
        return int.from_bytes(response[1:4], 'big'), response[0]

    def get_current_time(self):
        """Retrieves the current hardware clock in ticks."""
        self._send_command("GET_TIME")
        response = self.ser.read(3)
        return int.from_bytes(response, 'big') if len(response) == 3 else None

    def stop(self):
        """Stops the sequence and clears scheduled fires."""
        self._send_command("STOP_RESET")

    def hard_reset(self):
        """Clears all memory, registers, and sequences."""
        self._send_command("HARD_RESET")
        self.step_index = 0

    def close(self):
        if self.ser:
            self.ser.close()

    @staticmethod
    def to_24bit(value: float) -> int:
        """Standardizes any numeric value into the 0 to 2^24-1 range."""
        return int(value) & TimingBox.MAX_TICKS

    @staticmethod
    def get_tick_diff(t_future: int, t_now: int) -> int:
        """Calculates positive distance in modular 24-bit space."""
        return (t_future - t_now) & TimingBox.MAX_TICKS

    @staticmethod
    def is_future_tick(target: int, current: int) -> bool:
        """Determines if a target is ahead of the current clock using half-range logic."""
        return TimingBox.get_tick_diff(target, current) < TimingBox.HALF_RANGE

if __name__ == "__main__":
    
    box = TimingBox(port= Config.TimingBox.TEST_PORT) 
    box.connect()

    try:
        print("\nConfiguring Pin Mappings...")
        box.map_pin(physical_pin=10, logical_pin=0, invert=True)
        mapping = box.get_pin_mapping(10)
        print(f"Result: Physical 10 -> Logical {mapping[0]} (Inverted: {mapping[1]})")

        print("\nUploading Pianola Sequence...")
        box.add_step(logical_pins=[0], duration_ticks=TimingBox.to_24bit(0.5 / box.TICK_SEC))  # 0.5 seconds in ticks
        box.add_step(logical_pins=[1], duration_ticks=TimingBox.to_24bit(0.5 / box.TICK_SEC))  # 0.5 seconds in ticks

        box.finalize_sequence(repeat=False)
        print("Sequence uploaded and finalized.")

        print("\nTesting Immediate Execution (RUN_NOW)...")
        start_tick = box.run_now()
        print(f"Manual trigger started at tick: {start_tick}")
        time.sleep(1.2) # Wait for sequence to finish

        print("\nTesting Scheduled Execution (FIRE_AT)...")
        
        current_time = box.get_current_time()
        delay_ticks = TimingBox.to_24bit(2.0 / box.TICK_SEC)  # Schedule for 2 seconds in the future
        future_tick = (current_time + delay_ticks) & 0xFFFFFF
        
        print(f"Current Tick: {current_time}")
        print(f"Scheduling fire for tick: {future_tick} (~2.0s from now)")
        
        trig_time, success = box.fire_at(future_tick)
        
        if success:
            print(f"✔ Emulator acknowledged the future fire time at tick: {trig_time}.")
            print("Waiting for trigger... (Watch the Emulator console)")
            
            time.sleep(3.0)
        else:
            print("✘ Emulator rejected the fire time (indicated it is in the past).")

    except Exception as e:
        print(f"An error occurred during testing: {e}")

    finally:
        print("\n[CLEANUP] Stopping and resetting hardware...")
        box.stop()
        box.hard_reset()
        box.close()
        print("Test suite complete.")