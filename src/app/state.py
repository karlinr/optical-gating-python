import queue
import threading
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class UIEvent:
    topic: str
    payload: Any = None

class AppState:
    def __init__(self):
        self.event_queue = queue.Queue()
        self._lock = threading.Lock()
        self.latest_frame = None

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