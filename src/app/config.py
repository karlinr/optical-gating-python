from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path

class ExperimentConfig:
    # Paths for saving data and logs
    PARENT_ROOT = Path(__file__).parents[2]
    DATA_ROOT = PARENT_ROOT / "data"
    EXPERIMENT_NAME = "default_experiment"
    EXPERIMENT_DATA_PATH = f"{DATA_ROOT}/{EXPERIMENT_NAME}"

    # Performance settings
    BRIGHTFIELD_CHUNK_SIZE = 100
    FLUORESCENCE_CHUNK_SIZE = 1
    NUM_THREADS = 6

    # Logging settings
    LOGGING_LEVEL = "INFO"
    SAVE_BRIGHTFIELD_FRAMES = False
    SAVE_FLUORESCENCE_FRAMES = True

# Timing box and pin mapping
class TimingBox:
    PORT = 'COM7'#'COM5'#'COM3'
    EMULATOR_PORT = 'COM4'#'COM6'
    
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
    downsample: str
    roi: Optional[Tuple[int, int, int, int]]
    trigger_pin: int  # GPIO pin on the camera
    framerate: Optional[int] = None
    sensor_taps: int = None
    label: str = ""
    box_pins: List[int] = field(default_factory=list)

# Brightfield
class Cameras:
    BF = CameraConfig(
        label="Brightfield camera",
        serial="28600723",
        exposure_us=2000,
        gain=0.0,
        downsample="XI_DWN_2x2",
        roi=(828, 418, 484, 488),
        trigger_pin=2, # Physical GPIO pin on the camera
        box_pins=[
            TimingBox.Logical.BF
            ],
        framerate=80,
        sensor_taps = "XI_TAP_CNT_4"
    )

    # Fluorescence
    FL = CameraConfig(
            label="Fluorescence camera",
            serial="CEMAU2502004",
            exposure_us=3000,
            gain=1.0,
            downsample=None,
            roi=None,
            trigger_pin=3, # Physical GPIO pin on the camera
            box_pins=[
                TimingBox.Logical.FL_1, 
                TimingBox.Logical.LAS_BLUE, 
                TimingBox.Logical.LAS_GREEN
            ],
            framerate=None,
            sensor_taps = None
        )

class Gating:
    # Methods to estimate and predict phase
    # Options for PHASE_SOURCE: SAD or MLE
    # Options for PREDICTION_METHOD: BARRIER or KALMAN
    PHASE_SOURCE = "MLE_ANOMALY"
    PREDICTION_METHOD = "KALMAN"

    # Whether we should log all phase estimates or just the PHASE_SOURCE one
    ENABLED_ESTIMATORS = ["SAD", "MLE_ANOMALY"]

    # SAD parameters
    NUM_EXTRA_REF_FRAMES = 2
    MIN_PERIOD = 5
    LOWER_THRESHOLD_FACTOR = 0.5
    UPPER_THRESHOLD_FACTOR = 0.75
    MIN_HEART_RATE_HZ = 0.5

    # MLE parameters
    MLE_BOOTSTRAP_FRAMES = 1000
    MLE_BINS = 46
    MLE_MIN_NOISE = 1
    MLE_FIT_POINTS = 1

    # Barrier prediction parameters
    PHASE_HISTORY_LENGTH = 100
    MIN_FRAMES_FOR_PREDICTION = 3
    MAX_FRAMES_FOR_PREDICTION = 32
    MIN_HISTORY_FOR_PREDICTION = 50

    # Kalman filter parameters
    KALMAN_MEASUREMENT_NOISE = 0.0001
    KALMAN_PROCESS_NOISE = 0.0001

    # Prediction parameters
    PREDICTION_LATENCY = 0.05
    EXTRAPOLATION_FACTOR = 1.5

class Config:
    EMULATE_CAMERA = True  # Whether to use the camera emulator or real hardware

    ExperimentConfig = ExperimentConfig
    TimingBox = TimingBox
    Cameras = Cameras
    Gating = Gating