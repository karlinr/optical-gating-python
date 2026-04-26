from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class CameraConfig:
    """Template for camera-specific settings."""
    serial: str
    exposure_us: int
    gain: float
    downsample: int
    roi: Optional[Tuple[int, int, int, int]]
    trigger_pin: int  # GPIO pin on the camera
    framerate: Optional[int] = None
    label: str = ""
    box_pins: List[int] = field(default_factory=list)

class Config:
    # --- Hardware & Pin Mapping ---
    class Hardware:
        PORT = 'COM4'
        EMULATOR_PORT = 'COM5'
        TEST_PORT = 'COM6'
        
        class Physical(IntEnum):
            """Actual BNC ports on the Timing Box."""
            BNC_1 = 1
            FL_1 = 2
            FL_2 = 6
            BF = 4
            LAS_GREEN = 5
            LAS_BLUE = 7

        class Logical(IntEnum):
            """Software bits (0-31) for command packets."""
            BNC_1 = 0
            FL_1 = 1
            BF = 2
            LAS_GREEN = 3
            FL_2 = 4
            LAS_BLUE = 5

    # --- Camera Instances ---
    # Brightfield
    BF = CameraConfig(
        label="Brightfield camera",
        serial="28600723",
        exposure_us=1000,
        gain=0.0,
        downsample=1,
        roi=(856, 572, 484, 488),
        trigger_pin=2,
        box_pins=[
            Hardware.Logical.BF
            ],
        framerate=80
    )

    # Fluorescence
    FL = CameraConfig(
            label="Fluorescence camera",
            serial="CEMAU2502004",
            exposure_us=1000,
            gain=1.0,
            downsample=1,
            roi=None,
            trigger_pin=3,
            box_pins=[
                Hardware.Logical.FL_1, 
                Hardware.Logical.LAS_BLUE, 
                Hardware.Logical.LAS_GREEN
            ]
        )
    
    class Gating:
        PHASE_SOURCE = "SAD"

        LOG_ALL = True

        REFERENCE_LENGTH = 1000
        BOOTSTRAP_FRAMES = 1000
        N_BINS = 100