import queue
import threading
from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum, auto

@dataclass
class UIEvent:
    topic: str
    payload: Any = None

class ExperimentState(Enum):
    SYSTEM_IDLE = auto()
    CAMERA_PREVIEW = auto()
    CALIBRATING = auto()
    READY = auto()
    RUNNING_EXPERIMENT = auto()
    SYSTEM_ERROR = auto()

class AppState:
    def __init__(self):
        self.event_queue = queue.Queue()
        self._lock = threading.Lock()
        self.latest_frame = None
        self._current_state = ExperimentState.SYSTEM_IDLE

    def update_frame(self, frame):
        with self._lock:
            self.latest_frame = frame

    def get_latest_frame(self):
        with self._lock:
            return self.latest_frame
        
    def send_event(self, topic, payload=None):
        event = UIEvent(topic=topic, payload=payload)
        self.event_queue.put(event)

    def get_next_event(self):
        try:
            return self.event_queue.get_nowait()
        except queue.Empty:
            return None
        
    def set_state(self, new_state):
        with self._lock:
            self._current_state = new_state

        self.send_event("STATE_CHANGED", new_state.name)

    def get_state(self):
        with self._lock:
            return self._current_state