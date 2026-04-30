from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Timing box and pin mapping
class TimingBox:
    PORT = 'COM6'
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
        """Software bits (0-7) for command packets."""
        BNC_1 = 0
        FL_1 = 1
        BF = 2
        LAS_GREEN = 3
        FL_2 = 4
        LAS_BLUE = 5

# Cameras
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

# Brightfield
class Cameras:
    BF = CameraConfig(
        label="Brightfield camera",
        serial="28600723",
        exposure_us=1000,
        gain=0.0,
        downsample=1,
        roi=(856, 572, 484, 488),
        trigger_pin=2, # Physical GPIO pin on the camera
        box_pins=[
            TimingBox.Logical.BF
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
            trigger_pin=3, # Physical GPIO pin on the camera
            box_pins=[
                TimingBox.Logical.FL_1, 
                TimingBox.Logical.LAS_BLUE, 
                TimingBox.Logical.LAS_GREEN
            ]
        )

class Gating:
    # Methods to estimate and predict phase
    # Options for PHASE_SOURCE: SAD or MLE
    # Options for PREDICTION_METHOD: BARRIER or KALMAN
    PHASE_SOURCE = "SAD"
    PREDICTION_METHOD = "BARRIER"

    # Whether we should log all phase estimates or just the PHASE_SOURCE one
    LOG_ALL = True

    # SAD parameters
    MIN_PERIOD = 5
    LOWER_THRESHOLD_FACTOR = 0.5
    UPPER_THRESHOLD_FACTOR = 0.75
    MIN_HEART_RATE_HZ = 0.5

    # MLE parameters
    MLE_BOOTSTRAP_FRAMES = 2000
    MLE_BINS = 40
    MLE_MIN_NOISE = 1
    MLE_FIT_POINTS = 5

    # Barrier prediction parameters
    PHASE_HISTORY_LENGTH = 100
    MIN_FRAMES_FOR_PREDICTION = 10
    MIN_HISTORY_FOR_PREDICTION = 50

class Config:
    EMULATE_CAMERA = True  # Whether to use the camera emulator or real hardware

    TimingBox = TimingBox
    Cameras = Cameras
    Gating = Gating