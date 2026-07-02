# NOTE: This was (clearly) written using AI
# This is purely for testing.
# I may at some point attempt to write something myself but for now this will suffice.

import sys
import os
import time
import pickle
import queue
import multiprocessing
import numpy as np
from datetime import datetime
from loguru import logger
from collections import deque

from PyQt5.QtCore import QTimer, Qt, QPointF
from PyQt5.QtGui import QFont, QPainter, QPen, QColor, QBrush, QFontDatabase, QPalette
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QTabWidget, QLabel, QPushButton,
    QGroupBox, QComboBox, QCheckBox, QFrame, QSizePolicy, QDoubleSpinBox, QSpinBox
)
import pyqtgraph as pg

# Enforce system path awareness for project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import Config
from app.data_manager import data_manager
from interfaces.system import SystemController
from logic.phase_estimator import PhaseManager
from logic.predictors.base import predictor_registry
from logic.trigger_decider import TriggerDecider
from logic.estimators.base import estimator_registry

# ==============================================================================
# DESIGN TOKENS
# ==============================================================================
COLOR_VOID = "#0a0b10"
COLOR_PANEL = "#131520"
COLOR_PANEL_RAISED = "#1a1d2c"
COLOR_RECESSED = "#0d0e15"
COLOR_LINE = "#222636"
COLOR_LINE_BRIGHT = "#31374d"
COLOR_TEXT_PRIMARY = "#f3f4f6"
COLOR_TEXT_DIM = "#94a3b8"
COLOR_TEXT_FAINT = "#4b5563"

COLOR_PHOSPHOR = "#0dd5b1"     # primary instrument green-teal
COLOR_PHOSPHOR_DIM = "#064e43"
COLOR_AMBER = "#f59e0b"        # warning / lookahead tracking
COLOR_ROSE = "#f43f5e"         # stop / fault tracking
COLOR_VIOLET = "#8b5cf6"       # secondary channel tracker
COLOR_BLUE = "#3b82f6"         # timeline framing accent
COLOR_GREEN_LAMP = "#10b981"   # running state notification lamp

SERIES_COLORS = [COLOR_PHOSPHOR, COLOR_VIOLET, COLOR_AMBER, COLOR_BLUE, COLOR_ROSE]

FONT_MONO_CANDIDATES = ["JetBrains Mono", "Roboto Mono", "Consolas", "Menlo", "monospace"]
FONT_UI_CANDIDATES = ["Inter", "Segoe UI", "-apple-system", "Roboto", "sans-serif"]
FONT_MONO_STACK = ", ".join(f"'{f}'" if " " in f else f for f in FONT_MONO_CANDIDATES)
FONT_UI_STACK = ", ".join(f"'{f}'" if " " in f else f for f in FONT_UI_CANDIDATES)

MODERN_QSS = f"""
QMainWindow {{
    background-color: {COLOR_VOID};
}}
QWidget {{
    color: {COLOR_TEXT_PRIMARY};
    font-family: {FONT_UI_STACK};
    font-size: 11.5px;
}}
QToolTip {{
    background-color: {COLOR_PANEL_RAISED};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_LINE_BRIGHT};
    padding: 4px 6px;
}}
QTabWidget::pane {{
    border: 1px solid {COLOR_LINE};
    background-color: {COLOR_PANEL};
    border-radius: 6px;
    top: -1px;
}}
QTabWidget::tab-bar {{
    alignment: left;
}}
QTabBar::tab {{
    background-color: transparent;
    border: none;
    padding: 10px 18px;
    margin-right: 4px;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.5px;
    color: {COLOR_TEXT_DIM};
}}
QTabBar::tab:selected {{
    color: {COLOR_PHOSPHOR};
    border-bottom: 2px solid {COLOR_PHOSPHOR};
}}
QTabBar::tab:hover:not(:selected) {{
    color: {COLOR_TEXT_PRIMARY};
}}
QGroupBox {{
    border: 1px solid {COLOR_LINE};
    border-radius: 6px;
    margin-top: 14px;
    font-weight: 700;
    font-size: 10.5px;
    letter-spacing: 0.8px;
    color: {COLOR_TEXT_DIM};
    background-color: {COLOR_PANEL};
    padding: 12px 10px 10px 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    top: 2px;
    padding: 0 4px;
}}

/* Platform-agnostic layout overrides targeting drop-down lists */
QComboBox {{
    border: 1px solid {COLOR_LINE_BRIGHT};
    border-radius: 4px;
    padding: 4px 8px;
    background-color: {COLOR_RECESSED} !important;
    color: {COLOR_TEXT_PRIMARY} !important;
    min-height: 22px;
}}
QComboBox:!editable, QComboBox::drop-down:!editable {{
    background-color: {COLOR_RECESSED} !important;
}}
QComboBox:hover {{
    border-color: {COLOR_PHOSPHOR};
}}
QComboBox::drop-down {{
    border: none;
    background: transparent;
    width: 20px;
}}
QComboBox::arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {COLOR_TEXT_DIM};
    width: 0px;
    height: 0px;
    margin-right: 6px;
}}
QComboBox::arrow:hover {{
    border-top-color: {COLOR_PHOSPHOR};
}}
QComboBox QAbstractItemView {{
    background-color: {COLOR_PANEL_RAISED} !important;
    border: 1px solid {COLOR_LINE_BRIGHT};
    border-radius: 4px;
    outline: none;
    padding: 4px;
    color: {COLOR_TEXT_PRIMARY} !important;
}}

/* Spinbox styling. Note: under the Fusion style these widgets paint from the
   QPalette, so the dark application palette (build_dark_palette) is what makes
   the field background dark and the text light. The rules below add the border,
   corners and selection accents on top of that. */
QAbstractSpinBox {{
    background-color: {COLOR_RECESSED};
    color: {COLOR_TEXT_PRIMARY};
    selection-background-color: {COLOR_PHOSPHOR_DIM};
    selection-color: {COLOR_TEXT_PRIMARY};
    font-family: {FONT_MONO_STACK};
    min-height: 22px;
    border: 1px solid {COLOR_LINE_BRIGHT};
    border-radius: 4px;
    padding-right: 14px;
}}
QAbstractSpinBox:focus, QDoubleSpinBox:hover, QSpinBox:hover {{
    border-color: {COLOR_PHOSPHOR};
}}
QAbstractSpinBox:disabled {{
    color: {COLOR_TEXT_FAINT};
    background-color: {COLOR_PANEL};
    border-color: {COLOR_LINE};
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {{
    background: transparent;
    border: none;
    width: 14px;
}}
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-bottom: 4px solid {COLOR_TEXT_DIM};
}}
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid {COLOR_TEXT_DIM};
}}

QCheckBox {{
    spacing: 8px;
    color: {COLOR_TEXT_PRIMARY};
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {COLOR_LINE_BRIGHT};
    border-radius: 3px;
    background-color: {COLOR_RECESSED};
}}
QCheckBox::indicator:hover {{
    border-color: {COLOR_PHOSPHOR};
}}
QCheckBox::indicator:checked {{
    background-color: {COLOR_PHOSPHOR};
    border-color: {COLOR_PHOSPHOR};
}}
QPushButton {{
    border-radius: 4px;
    padding: 8px 14px;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.5px;
    border: 1px solid transparent;
}}
QPushButton:disabled {{
    background-color: {COLOR_PANEL_RAISED} !important;
    color: {COLOR_TEXT_FAINT} !important;
    border: 1px solid {COLOR_LINE} !important;
}}
QFrame#ViewportContainer {{
    border: 1px solid {COLOR_LINE};
    border-radius: 6px;
    background-color: {COLOR_VOID};
    padding: 2px;
}}
QFrame#Hairline {{
    background-color: {COLOR_LINE};
    max-height: 1px;
    min-height: 1px;
}}
QLabel#ReadoutValue {{
    font-family: {FONT_MONO_STACK};
    color: {COLOR_TEXT_PRIMARY};
    font-size: 13px;
    font-weight: 600;
}}
QLabel#ReadoutCaption {{
    color: {COLOR_TEXT_DIM};
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.8px;
}}
QLabel#ViewportTitle {{
    color: {COLOR_TEXT_DIM};
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.8px;
}}
"""


def build_dark_palette():
    """A fully dark QPalette for the Fusion style.

    Qt's Fusion style draws input widgets (spin boxes, line edits, combo popups)
    primarily from the palette rather than the stylesheet. Without this, those
    fields fall back to Fusion's light defaults and render with a pale background
    and washed-out text. Setting the palette application-wide is what guarantees
    the numeric textboxes get a dark background and bright, readable text.
    """
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLOR_VOID))
    palette.setColor(QPalette.WindowText, QColor(COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.Base, QColor(COLOR_RECESSED))
    palette.setColor(QPalette.AlternateBase, QColor(COLOR_PANEL))
    palette.setColor(QPalette.ToolTipBase, QColor(COLOR_PANEL_RAISED))
    palette.setColor(QPalette.ToolTipText, QColor(COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.Text, QColor(COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.PlaceholderText, QColor(COLOR_TEXT_FAINT))
    palette.setColor(QPalette.Button, QColor(COLOR_PANEL_RAISED))
    palette.setColor(QPalette.ButtonText, QColor(COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.BrightText, QColor(COLOR_ROSE))
    palette.setColor(QPalette.Link, QColor(COLOR_BLUE))
    palette.setColor(QPalette.Highlight, QColor(COLOR_PHOSPHOR_DIM))
    palette.setColor(QPalette.HighlightedText, QColor(COLOR_TEXT_PRIMARY))

    # Disabled-state colors so greyed-out controls stay legible and on-theme.
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(COLOR_TEXT_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(COLOR_TEXT_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(COLOR_TEXT_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.Base, QColor(COLOR_PANEL))
    palette.setColor(QPalette.Disabled, QPalette.Button, QColor(COLOR_PANEL))
    return palette


def run_hardware_loop(ui_queue, bf_queue, fl_queue, res_queue, control_queue, stop_event, runtime_config):
    """Independent backend process driving the real-time hardware loop."""
    Config.Gating.PHASE_SOURCE = runtime_config["PHASE_SOURCE"]
    Config.Gating.PREDICTION_METHOD = runtime_config["PREDICTION_METHOD"]
    Config.ExperimentConfig.SAVE_BRIGHTFIELD_FRAMES = runtime_config["SAVE_BF"]
    Config.ExperimentConfig.SAVE_FLUORESCENCE_FRAMES = runtime_config["SAVE_FL"]
    Config.Gating.ENABLED_ESTIMATORS = runtime_config["ENABLED_ESTIMATORS"]
    
    Config.Gating.KALMAN_MEASUREMENT_NOISE = runtime_config["PARAM_K_MEAS"]
    Config.Gating.KALMAN_PROCESS_NOISE = runtime_config["PARAM_K_PROC"]
    Config.Gating.MLE_MODEL_BOOTSTRAP_FRAMES = runtime_config["PARAM_BOOTSTRAP"]

    storage_path = Config.ExperimentConfig.EXPERIMENT_DATA_PATH
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    storage_path = f"{storage_path}/dashboard_run_{timestamp_str}"

    logger.remove()
    logger.add(sys.stderr, level=Config.ExperimentConfig.LOGGING_LEVEL,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", enqueue=True)
    os.makedirs(f"{storage_path}/logs", exist_ok=True)
    logger.add(f"{storage_path}/logs/experiment.log", rotation="10 MB", level=Config.ExperimentConfig.LOGGING_LEVEL, retention="10 days", enqueue=True)

    hist_timestamps = []
    hist_framerates = []
    hist_phase_results = []
    hist_prediction_results = []
    hist_committed_triggers = []

    try:
        with SystemController() as controller:
            bf_shape, fl_shape = controller.connect_all()
            logger.info("All hardware components connected successfully.")

            controller.synchronise_camera()
            controller.setup_cameras_for_experiment()
            controller.setup_timing_box_for_experiment()
            data_manager.configure(storage_path)

            phase_manager = PhaseManager()
            pred_method = Config.Gating.PREDICTION_METHOD
            if pred_method in predictor_registry:
                phase_predictor = predictor_registry[pred_method]()
            else:
                raise ValueError(f"Unsupported prediction method: {pred_method}")

            trigger_controller = TriggerDecider()
            iterations = Config.ExperimentConfig.ITERATIONS
            est_names = []

            batch_data = []
            batch_size = 10  

            for i in range(iterations):
                if stop_event.is_set():
                    break

                while not control_queue.empty():
                    try:
                        msg = control_queue.get_nowait()
                        if msg.get("type") == "SET_RESIDUAL_SOURCE":
                            runtime_config["RESIDUAL_SOURCE"] = msg.get("value")
                    except queue.Empty:
                        break

                frame, timestamp, metadata = controller.get_latest_bf_frame()
                if Config.ExperimentConfig.SAVE_BRIGHTFIELD_FRAMES:
                    data_manager.save("brightfield", frame.copy(), chunk_size=Config.ExperimentConfig.BRIGHTFIELD_CHUNK_SIZE)

                try:
                    bf_queue.put_nowait(frame)
                except queue.Full:
                    pass

                phase_results = phase_manager.update(frame, timestamp=timestamp)
                active = phase_results.get("ACTIVE", {})
                current_status = active.get("status", "INITIALIZING")

                if not est_names:
                    est_names = sorted(list(set(name for name in phase_results if name != "ACTIVE")))

                if current_status == "READY":
                    target_res_source = runtime_config["RESIDUAL_SOURCE"]
                    residual_frame = None
                    
                    if target_res_source in phase_results and "residual" in phase_results[target_res_source]:
                        residual_frame = phase_results[target_res_source]["residual"]

                    if residual_frame is not None:
                        try:
                            res_queue.put_nowait(residual_frame)
                        except queue.Full:
                            pass

                hist_timestamps.append(timestamp)
                current_fps = metadata.get("framerate", 0.0)
                hist_framerates.append(current_fps)

                ui_data = {
                    "iteration": i,
                    "status": current_status,
                    "timestamp": timestamp,
                    "framerate": current_fps,
                    "active_phase": active.get("phase"),
                    "est_period": None,
                    "lookahead": None,
                    "k_phase": None,
                    "k_velocity": None,
                    "estimators": {},
                    "trigger": None
                }

                for name in est_names:
                    if name not in Config.Gating.ENABLED_ESTIMATORS:
                        continue
                    est_res = phase_results.get(name, {})
                    metrics = est_res.get("metrics", {})
                    probs = est_res.get("probabilities", {})
                    p_bins = probs.get("phase_bins")
                    b_idx = metrics.get("best_index")

                    ui_data["estimators"][name] = {
                        "phase": est_res.get("phase"),
                        "reduced_chi_squared": metrics.get("reduced_chi_squared"),
                        "sad_score": metrics.get("sad_score"),
                        "uncertainty": metrics.get("uncertainty_estimate"),
                        "scores": metrics.get("scores"),
                        "drift_x": metrics.get("drift_x"),
                        "drift_y": metrics.get("drift_y"),
                        "vertex_offset": metrics.get("vertex_offset"),
                        "best_index": b_idx,
                        "reference_period": metrics.get("reference_period"),
                        "unknown_anomaly": probs.get("unknown_anomaly"),
                        "best_bin_prob": p_bins[b_idx] if (p_bins is not None and b_idx is not None and b_idx < len(p_bins)) else None
                    }

                predicted_time_rel = None
                pred_metadata = {}

                if active.get("status") == "READY":
                    current_phase = active["phase"]
                    target_phase = active["target_phase"]
                    barrier_phase = active["barrier_phase"]
                    active_metrics = active.get("metrics", {})

                    phase_predictor.update_phase(current_phase, timestamp, **active_metrics)
                    predicted_time_rel, pred_metadata = phase_predictor.predict_target_time(target_phase, barrier_phase=barrier_phase, **active_metrics)

                    if predicted_time_rel is not None:
                        est_period = pred_metadata["est_period"]
                        absolute_predicted_time = timestamp + predicted_time_rel

                        fire_signal, relative_wait = trigger_controller.evaluate_trigger(timestamp, absolute_predicted_time, est_period)

                        if fire_signal:
                            exact_hardware_target = timestamp + relative_wait
                            box_time, response = controller.trigger_fl_frame(exact_hardware_target)
                            if response == 1:
                                logger.debug("Fluorescence trigger successfully committed to hardware.")
                                hist_committed_triggers.append((timestamp, exact_hardware_target))
                                ui_data["trigger"] = (timestamp, exact_hardware_target)

                                def async_fluorescence_save(target=exact_hardware_target, fl_q=fl_queue):
                                    try:
                                        fl_frame, fl_timestamp, fl_metadata = controller.get_latest_fl_frame()
                                        if Config.ExperimentConfig.SAVE_FLUORESCENCE_FRAMES:
                                            data_manager.save("fluorescence", fl_frame, chunk_size=Config.ExperimentConfig.FLUORESCENCE_CHUNK_SIZE)
                                        try:
                                            fl_q.put_nowait(fl_frame)
                                        except queue.Full:
                                            pass
                                    except Exception as e:
                                        logger.error(f"Background fluorescence pipeline failed: {e}")

                                data_manager.submit_task(async_fluorescence_save)
                            else:
                                trigger_controller.handle_hardware_rejection(timestamp, est_period)

                if predicted_time_rel is not None:
                    hist_prediction_results.append((predicted_time_rel, pred_metadata))
                    ui_data["est_period"] = pred_metadata.get("est_period")
                    ui_data["lookahead"] = predicted_time_rel
                    ui_data["k_phase"] = pred_metadata.get("phase_estimate")
                    ui_data["k_velocity"] = pred_metadata.get("phase_velocity_estimate")
                else:
                    hist_prediction_results.append(None)

                for series_name in list(phase_results.keys()):
                    if isinstance(phase_results[series_name], dict):
                        phase_results[series_name].pop("residual", None)

                hist_phase_results.append(phase_results)

                batch_data.append(ui_data)
                if len(batch_data) >= batch_size:
                    try:
                        ui_queue.put_nowait(batch_data)
                    except Exception:
                        pass
                    batch_data = []

            if batch_data:
                try:
                    ui_queue.put_nowait(batch_data)
                except Exception:
                    pass

            metrics_save_path = os.path.join(storage_path, "metrics.pkl")
            with open(metrics_save_path, "wb") as f:
                pickle.dump({
                    "timestamps": hist_timestamps,
                    "framerates": hist_framerates,
                    "phase_results": hist_phase_results,
                    "prediction_results": hist_prediction_results,
                    "committed_triggers": hist_committed_triggers,
                }, f)

            ui_queue.put({"status_type": "finished", "storage_path": storage_path})

    except Exception as e:
        import traceback
        ui_queue.put({"status_type": "error", "message": f"{str(e)}\n{traceback.format_exc()}"})
    finally:
        data_manager.close()


class PhaseDialWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(96, 96)
        self.phase = None
        self.target_phase = None
        self.armed = False
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

    def set_phase(self, phase, target_phase=None, armed=False):
        self.phase = phase
        self.target_phase = target_phase
        self.armed = armed
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        side = min(self.width(), self.height())
        cx, cy = self.width() / 2.0, self.height() / 2.0
        r = side / 2.0 - 8

        painter.setPen(QPen(QColor(COLOR_LINE_BRIGHT), 1.5))
        painter.setBrush(QBrush(QColor(COLOR_RECESSED)))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        painter.setPen(QPen(QColor(COLOR_TEXT_FAINT), 1.2))
        for deg in range(0, 360, 30):
            import math
            rad = math.radians(deg - 90)
            inner = r - 6
            outer = r - 1
            x1, y1 = cx + inner * math.cos(rad), cy + inner * math.sin(rad)
            x2, y2 = cx + outer * math.cos(rad), cy + outer * math.sin(rad)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        if self.target_phase is not None:
            import math
            rad = math.radians(np.degrees(self.target_phase) - 90)
            tx, ty = cx + (r - 3) * math.cos(rad), cy + (r - 3) * math.sin(rad)
            pen_color = COLOR_AMBER if self.armed else COLOR_TEXT_FAINT
            painter.setPen(QPen(QColor(pen_color), 3))
            painter.drawLine(QPointF(cx, cy), QPointF(tx, ty))

        if self.phase is not None:
            import math
            rad = math.radians(np.degrees(self.phase) - 90)
            nx, ny = cx + (r - 10) * math.cos(rad), cy + (r - 10) * math.sin(rad)
            painter.setPen(QPen(QColor(COLOR_PHOSPHOR), 2.4))
            painter.drawLine(QPointF(cx, cy), QPointF(nx, ny))
            painter.setBrush(QBrush(QColor(COLOR_PHOSPHOR)))
            painter.setPen(QPen(QColor(COLOR_PHOSPHOR), 0))
            painter.drawEllipse(QPointF(cx, cy), 3.2, 3.2)
        else:
            painter.setBrush(QBrush(QColor(COLOR_TEXT_FAINT)))
            painter.setPen(QPen(QColor(COLOR_TEXT_FAINT), 0))
            painter.drawEllipse(QPointF(cx, cy), 2.6, 2.6)

        painter.end()


class StatusLamp(QWidget):
    def __init__(self, color=COLOR_TEXT_FAINT, parent=None):
        super().__init__(parent)
        self.setFixedSize(9, 9)
        self._color = color
        self._base_color = color
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._end_flash)

    def set_color(self, color):
        self._color = color
        self._base_color = color
        self.update()

    def flash(self, color=None, duration_ms=140):
        self._color = color or COLOR_AMBER
        self.update()
        self._flash_timer.start(duration_ms)

    def _end_flash(self):
        self._color = self._base_color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(self._color)))
        painter.drawEllipse(1, 1, 7, 7)
        painter.end()


class LiveDashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gated Acquisition Console")
        self.resize(1680, 960)
        self._load_fonts()
        self.setStyleSheet(MODERN_QSS)

        pg.setConfigOption('background', COLOR_PANEL)
        pg.setConfigOption('foreground', COLOR_TEXT_DIM)
        pg.setConfigOption('antialias', False)
        pg.setConfigOptions(useOpenGL=False)

        self.ui_queue = multiprocessing.Queue(maxsize=2000)
        self.bf_queue = multiprocessing.Queue(maxsize=1)
        self.fl_queue = multiprocessing.Queue(maxsize=1)
        self.res_queue = multiprocessing.Queue(maxsize=1)
        self.control_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.worker = None

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(lambda: self.refresh_ui_plots(force_replot=False))

        self.ui_window = 500
        self.ui_timestamps = deque(maxlen=self.ui_window)
        self.ui_framerates = deque(maxlen=self.ui_window)
        self.ui_active_phases = deque(maxlen=self.ui_window)
        self.ui_periods = deque(maxlen=self.ui_window)
        self.ui_lookaheads = deque(maxlen=self.ui_window)
        self.ui_k_phases = deque(maxlen=self.ui_window)
        self.ui_k_velocities = deque(maxlen=self.ui_window)

        self.est_names = []
        self.ui_est_phases = {}
        self.ui_chi_squares = {}
        self.ui_drift_xs = {}
        self.ui_drift_ys = {}
        self.ui_p_anons = {}
        self.ui_p_bests = {}
        
        self.ui_uncertainties = {}
        self.ui_sad_scores = {}

        self.raw_vertex_offsets = {}
        self.raw_frac_xs = {}
        self.raw_frac_ys = {}
        self.raw_phases_chunk = {}
        self.ui_triggers_x = deque(maxlen=200)
        self.ui_triggers_y = deque(maxlen=200)
        
        self.latest_scores = {}
        self.latest_best_idx = {}
        self.latest_v_offset = {}
        self.latest_ref_period = {}

        self._bf_level_counter = 0
        self._last_seen_status = None

        self.init_interface_layout()
        self._populate_dynamic_registries()
        
        self.active_backend_source = self.combo_source.currentText()
        self.tabs.currentChanged.connect(self.on_tab_changed)

    def _load_fonts(self):
        font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
        if os.path.isdir(font_dir):
            for fname in os.listdir(font_dir):
                if fname.lower().endswith((".ttf", ".otf")):
                    QFontDatabase.addApplicationFont(os.path.join(font_dir, fname))

    def _populate_dynamic_registries(self):
        estimators = sorted(list(estimator_registry.keys()))
        predictors = sorted(list(predictor_registry.keys()))

        self.combo_source.clear()
        self.combo_source.addItems(estimators)
        if Config.Gating.PHASE_SOURCE in estimators:
            self.combo_source.setCurrentText(Config.Gating.PHASE_SOURCE)

        self.combo_predictor.clear()
        self.combo_predictor.addItems(predictors)
        if Config.Gating.PREDICTION_METHOD in predictors:
            self.combo_predictor.setCurrentText(Config.Gating.PREDICTION_METHOD)

        self.combo_res_source.clear()
        self.combo_res_source.addItems(estimators)
        if Config.Gating.PHASE_SOURCE in estimators:
            self.combo_res_source.setCurrentText(Config.Gating.PHASE_SOURCE)

        for i in reversed(range(self.methods_layout.count())): 
            widget = self.methods_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        self.estimator_checkboxes = {}
        for est in estimators:
            chk = QCheckBox(est)
            chk.setChecked(est in Config.Gating.ENABLED_ESTIMATORS)
            self.methods_layout.addWidget(chk)
            self.estimator_checkboxes[est] = chk

    def _fix_combo_palette(self, combo):
        palette = combo.palette()
        palette.setColor(QPalette.Window, QColor(COLOR_RECESSED))
        palette.setColor(QPalette.Base, QColor(COLOR_RECESSED))
        palette.setColor(QPalette.Button, QColor(COLOR_RECESSED))
        palette.setColor(QPalette.Text, QColor(COLOR_TEXT_PRIMARY))
        palette.setColor(QPalette.ButtonText, QColor(COLOR_TEXT_PRIMARY))
        palette.setColor(QPalette.Highlight, QColor(COLOR_PHOSPHOR_DIM))
        palette.setColor(QPalette.HighlightedText, QColor(COLOR_TEXT_PRIMARY))
        combo.setPalette(palette)
        
        view = combo.view()
        if view:
            view.setStyleSheet(f"background-color: {COLOR_PANEL_RAISED}; color: {COLOR_TEXT_PRIMARY}; selection-background-color: {COLOR_PHOSPHOR_DIM}; border: 1px solid {COLOR_LINE_BRIGHT};")

    def _fix_spin_palette(self, spin):
        palette = spin.palette()
        palette.setColor(QPalette.Base, QColor(COLOR_RECESSED))
        palette.setColor(QPalette.Button, QColor(COLOR_RECESSED))
        palette.setColor(QPalette.Text, QColor(COLOR_TEXT_PRIMARY))
        palette.setColor(QPalette.ButtonText, QColor(COLOR_TEXT_PRIMARY))
        palette.setColor(QPalette.Highlight, QColor(COLOR_PHOSPHOR_DIM))
        palette.setColor(QPalette.HighlightedText, QColor(COLOR_TEXT_PRIMARY))
        spin.setPalette(palette)
        spin.setButtonSymbols(QDoubleSpinBox.UpDownArrows)
        spin.setAlignment(Qt.AlignVCenter)

    def init_interface_layout(self):
        main_central_widget = QWidget()
        self.setCentralWidget(main_central_widget)
        outer_layout = QHBoxLayout(main_central_widget)
        outer_layout.setContentsMargins(14, 14, 14, 14)
        outer_layout.setSpacing(14)

        left_panel = QVBoxLayout()
        left_panel.setSpacing(12)

        # --- Masthead ---
        masthead = QHBoxLayout()
        title_lbl = QLabel("GATED ACQUISITION CONSOLE")
        title_lbl.setStyleSheet(f"font-size: 13px; font-weight: 700; letter-spacing: 1.4px; color: {COLOR_TEXT_PRIMARY};")
        subtitle_lbl = QLabel("Phase-locked fluorescence triggering")
        subtitle_lbl.setStyleSheet(f"font-size: 10.5px; color: {COLOR_TEXT_DIM};")
        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title_block.addWidget(title_lbl)
        title_block.addWidget(subtitle_lbl)
        masthead.addLayout(title_block)
        masthead.addStretch()
        left_panel.addLayout(masthead)

        # Executive Control Buttons
        cmd_box = QGroupBox("EXECUTIVE CONTROL")
        cmd_layout = QHBoxLayout(cmd_box)
        cmd_layout.setSpacing(10)
        self.btn_start = QPushButton("\u25b6  Start Acquisition")
        self.btn_start.setStyleSheet(
            f"QPushButton {{ background-color: {COLOR_PHOSPHOR}; color: #06231f; }}"
            f"QPushButton:hover {{ background-color: #16e6c0; }}"
            f"QPushButton:pressed {{ background-color: #0bb89a; }}"
        )
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self.start_acquisition)

        self.btn_stop = QPushButton("\u25a0  Emergency Stop")
        self.btn_stop.setStyleSheet(
            f"QPushButton {{ background-color: {COLOR_ROSE}; color: #2a0a0e; }}"
            f"QPushButton:hover {{ background-color: #fb5870; }}"
            f"QPushButton:pressed {{ background-color: #d62d47; }}"
        )
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_acquisition)

        cmd_layout.addWidget(self.btn_start)
        cmd_layout.addWidget(self.btn_stop)
        left_panel.addWidget(cmd_box)

        # Primary Configuration Settings Card
        settings_box = QGroupBox("INSTRUMENT PARAMETERS")
        settings_grid = QGridLayout(settings_box)
        settings_grid.setVerticalSpacing(10)
        settings_grid.setHorizontalSpacing(14)

        lbl_source = QLabel("Phase Source")
        lbl_source.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 11px;")
        settings_grid.addWidget(lbl_source, 0, 0)
        self.combo_source = QComboBox()
        self._fix_combo_palette(self.combo_source)
        settings_grid.addWidget(self.combo_source, 0, 1)

        lbl_pred = QLabel("Prediction Logic")
        lbl_pred.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 11px;")
        settings_grid.addWidget(lbl_pred, 0, 2)
        self.combo_predictor = QComboBox()
        self._fix_combo_palette(self.combo_predictor)
        settings_grid.addWidget(self.combo_predictor, 0, 3)

        lbl_methods = QLabel("Execute Methods")
        lbl_methods.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 11px;")
        settings_grid.addWidget(lbl_methods, 1, 0)
        
        self.methods_layout = QHBoxLayout()
        self.methods_layout.setSpacing(10)
        settings_grid.addLayout(self.methods_layout, 1, 1, 1, 3)

        # Advanced Engine Configurations Card
        lbl_meas = QLabel("Kalman Measurement Noise")
        lbl_meas.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 10.5px;")
        settings_grid.addWidget(lbl_meas, 2, 0)
        self.spin_k_meas = QDoubleSpinBox()
        self.spin_k_meas.setRange(1e-7, 1.0)
        self.spin_k_meas.setDecimals(6)
        self.spin_k_meas.setSingleStep(0.0001)
        self.spin_k_meas.setValue(Config.Gating.KALMAN_MEASUREMENT_NOISE)
        self._fix_spin_palette(self.spin_k_meas)
        settings_grid.addWidget(self.spin_k_meas, 2, 1)

        lbl_proc = QLabel("Kalman Process Noise")
        lbl_proc.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 10.5px;")
        settings_grid.addWidget(lbl_proc, 2, 2)
        self.spin_k_proc = QDoubleSpinBox()
        self.spin_k_proc.setRange(1e-8, 1.0)
        self.spin_k_proc.setDecimals(6)
        self.spin_k_proc.setSingleStep(0.00001)
        self.spin_k_proc.setValue(Config.Gating.KALMAN_PROCESS_NOISE)
        self._fix_spin_palette(self.spin_k_proc)
        settings_grid.addWidget(self.spin_k_proc, 2, 3)

        lbl_boot = QLabel("Model Bootstrap Frames")
        lbl_boot.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 10.5px;")
        settings_grid.addWidget(lbl_boot, 3, 0)
        self.spin_bootstrap = QSpinBox()
        self.spin_bootstrap.setRange(100, 10000)
        self.spin_bootstrap.setSingleStep(500)
        self.spin_bootstrap.setValue(Config.Gating.MLE_MODEL_BOOTSTRAP_FRAMES)
        self._fix_spin_palette(self.spin_bootstrap)
        settings_grid.addWidget(self.spin_bootstrap, 3, 1)

        self.chk_save_bf = QCheckBox("Record brightfield array")
        self.chk_save_bf.setChecked(Config.ExperimentConfig.SAVE_BRIGHTFIELD_FRAMES)
        settings_grid.addWidget(self.chk_save_bf, 4, 0, 1, 2)

        self.chk_save_fl = QCheckBox("Record fluorescence array")
        self.chk_save_fl.setChecked(Config.ExperimentConfig.SAVE_FLUORESCENCE_FRAMES)
        settings_grid.addWidget(self.chk_save_fl, 4, 2, 1, 2)
        left_panel.addWidget(settings_box)

        # Pipeline Status Card
        status_box = QGroupBox("PIPELINE STATUS")
        status_outer = QVBoxLayout(status_box)
        status_outer.setContentsMargins(4, 2, 4, 6)
        status_layout = QHBoxLayout()
        status_layout.setSpacing(0)
        status_outer.addLayout(status_layout)

        def add_divider():
            line = QFrame()
            line.setObjectName("Hairline")
            line.setFixedWidth(1)
            status_layout.addWidget(line)
            status_layout.addSpacing(18)

        lamp_block = QHBoxLayout()
        lamp_block.setSpacing(8)
        self.status_lamp = StatusLamp(COLOR_TEXT_FAINT)
        lamp_v = QVBoxLayout()
        lamp_v.addStretch()
        lamp_v.addWidget(self.status_lamp)
        lamp_v.addStretch()
        lamp_block.addLayout(lamp_v)

        def build_readout(caption):
            col = QVBoxLayout()
            col.setSpacing(3)
            value_lbl = QLabel("\u2014")
            value_lbl.setObjectName("ReadoutValue")
            cap_lbl = QLabel(caption)
            cap_lbl.setObjectName("ReadoutCaption")
            col.addWidget(value_lbl)
            col.addWidget(cap_lbl)
            return col, value_lbl

        col_status, self.lbl_status = build_readout("STATUS")
        self.lbl_status.setText("IDLE")
        lamp_block.addLayout(col_status)
        status_layout.addLayout(lamp_block)
        status_layout.addSpacing(18)
        add_divider()

        col_frame, self.lbl_frame = build_readout("FRAME")
        self.lbl_frame.setText("0")
        status_layout.addLayout(col_frame)
        status_layout.addSpacing(18)
        add_divider()

        col_fps, self.lbl_fps = build_readout("FRAMERATE (HZ)")
        self.lbl_fps.setText("0.0")
        status_layout.addLayout(col_fps)
        status_layout.addSpacing(18)
        add_divider()

        col_runtime, self.lbl_runtime = build_readout("RUN TIME")
        self.lbl_runtime.setText("00:00:00")
        status_layout.addLayout(col_runtime)
        status_layout.addSpacing(18)
        add_divider()

        col_triggers, self.lbl_triggers = build_readout("TRIGGERS FIRED")
        self.lbl_triggers.setText("0")
        status_layout.addLayout(col_triggers)

        status_layout.addStretch()
        add_divider()

        dial_well = QFrame()
        dial_well.setObjectName("DisplayPanel")
        dial_well_layout = QVBoxLayout(dial_well)
        dial_well_layout.setContentsMargins(10, 8, 10, 8)
        dial_well_layout.setSpacing(4)
        dial_well_layout.setAlignment(Qt.AlignHCenter)
        self.phase_dial = PhaseDialWidget()
        self.phase_dial.setMinimumSize(72, 72)
        dial_well_layout.addWidget(self.phase_dial, alignment=Qt.AlignHCenter)
        dial_caption = QLabel("CYCLIC PHASE")
        dial_caption.setObjectName("ReadoutCaption")
        dial_caption.setAlignment(Qt.AlignHCenter)
        dial_well_layout.addWidget(dial_caption)
        status_layout.addWidget(dial_well)

        left_panel.addWidget(status_box)

        self._run_start_time = None
        self._trigger_count = 0
        self.runtime_timer = QTimer()
        self.runtime_timer.timeout.connect(self._tick_runtime_label)

        # Viewports Configuration Container Box
        video_box = QGroupBox("OPTICAL VIEWPORTS")
        video_layout = QVBoxLayout(video_box)
        video_layout.setSpacing(8)

        def build_viewport_frame(title_text, widget, has_combo=False):
            container = QFrame()
            container.setObjectName("ViewportContainer")
            v_wrap = QVBoxLayout(container)
            v_wrap.setContentsMargins(4, 4, 4, 4)
            v_wrap.setSpacing(4)
            
            hdr = QHBoxLayout()
            lbl = QLabel(title_text)
            lbl.setObjectName("ViewportTitle")
            hdr.addWidget(lbl)
            hdr.addStretch()
            
            combo = None
            if has_combo:
                combo = QComboBox()
                self._fix_combo_palette(combo)
                combo.setMaximumHeight(20)
                combo.setStyleSheet(f"font-size: 10px; padding: 1px 4px; background-color: {COLOR_PANEL_RAISED};")
                hdr.addWidget(combo)
                
            v_wrap.addLayout(hdr)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            v_wrap.addWidget(widget)
            return container, combo

        self.bf_plot = pg.PlotWidget()
        self._configure_image_plot(self.bf_plot)
        self.bf_image_item = pg.ImageItem()
        self.bf_plot.addItem(self.bf_image_item)
        video_layout.addWidget(build_viewport_frame("BRIGHTFIELD \u00b7 STRUCTURAL CHANNEL", self.bf_plot)[0], stretch=1)

        self.res_plot = pg.PlotWidget()
        self._configure_image_plot(self.res_plot)
        self.res_image_item = pg.ImageItem()
        pos = np.array([0.0, 0.5, 1.0])
        color = np.array([[37, 99, 235, 255], [13, 14, 21, 255], [244, 63, 94, 255]], dtype=np.ubyte)
        self.res_image_item.setColorMap(pg.ColorMap(pos, color))
        self.res_plot.addItem(self.res_image_item)
        
        res_container, self.combo_res_source = build_viewport_frame("RESIDUAL MODEL DEVIATION", self.res_plot, has_combo=True)
        self.combo_res_source.currentTextChanged.connect(self.change_residual_source)
        video_layout.addWidget(res_container, stretch=1)

        self.fl_plot = pg.PlotWidget()
        self._configure_image_plot(self.fl_plot)
        self.fl_image_item = pg.ImageItem()
        self.fl_plot.addItem(self.fl_image_item)
        video_layout.addWidget(build_viewport_frame("FLUORESCENCE \u00b7 LAST COMMITTED CAPTURE", self.fl_plot)[0], stretch=1)

        left_panel.addWidget(video_box, stretch=4)
        outer_layout.addLayout(left_panel, stretch=1)

        # Tabbed Metrics Plot Panels
        self.tabs = QTabWidget()
        outer_layout.addWidget(self.tabs, stretch=2)

        self.tab_gating = QWidget()
        self.tab_phase = QWidget()
        self.tab_alignment = QWidget()
        self.tab_peak_locking = QWidget()
        self.tab_scores = QWidget()

        self.tabs.addTab(self.tab_gating, "GATING")
        self.tabs.addTab(self.tab_phase, "PHASE SPACE")
        self.tabs.addTab(self.tab_alignment, "MODEL BOUNDS")
        self.tabs.addTab(self.tab_peak_locking, "SUB-PIXEL NOISE")
        self.tabs.addTab(self.tab_scores, "SCORE CURVES")

        self.setup_tab_gating()
        self.setup_tab_phase()
        self.setup_tab_alignment()
        self.setup_tab_peak_locking()
        self.setup_tab_scores()

    def _configure_image_plot(self, plot_widget):
        plot_widget.setAspectLocked(True)
        plot_widget.invertY(True)
        plot_widget.setBackground(COLOR_RECESSED)
        plot_widget.getAxis('left').hide()
        plot_widget.getAxis('bottom').hide()
        plot_widget.getAxis('left').setStyle(tickLength=0)
        plot_widget.getAxis('bottom').setStyle(tickLength=0)
        plot_widget.setMenuEnabled(False)

    def _style_plot(self, plot_widget, title):
        plot_widget.setTitle(title, color=COLOR_TEXT_DIM, size="10.5pt")
        plot_widget.showGrid(x=True, y=True, alpha=0.08)
        plot_widget.getAxis('left').setPen(pg.mkPen(COLOR_LINE_BRIGHT))
        plot_widget.getAxis('bottom').setPen(pg.mkPen(COLOR_LINE_BRIGHT))
        plot_widget.getAxis('left').setTextPen(pg.mkPen(COLOR_TEXT_DIM))
        plot_widget.getAxis('bottom').setTextPen(pg.mkPen(COLOR_TEXT_DIM))
        plot_widget.setStyleSheet(f"border: 1px solid {COLOR_LINE}; border-radius: 6px;")
        plot_item = plot_widget.getPlotItem()
        plot_item.setDownsampling(mode='peak')
        plot_item.setClipToView(True)
        plot_widget.setAntialiasing(False)

    def setup_tab_gating(self):
        layout = QGridLayout(self.tab_gating)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        self.p0 = pg.PlotWidget()
        self._style_plot(self.p0, "Camera Framerate (fps)")
        self.curve_framerate = self.p0.plot(pen=pg.mkPen(COLOR_BLUE, width=1.6))
        layout.addWidget(self.p0, 0, 0)

        self.p4 = pg.PlotWidget()
        self._style_plot(self.p4, "Estimated Cyclic Period (s)")
        self.curve_period = self.p4.plot(pen=pg.mkPen(COLOR_VIOLET, width=1.6))
        layout.addWidget(self.p4, 0, 1)

        self.p5 = pg.PlotWidget()
        self._style_plot(self.p5, "Gated Lookahead and Hardware Trigger Commitments")
        self.p5.addLegend()
        self.scatter_lookahead = pg.ScatterPlotItem(size=3, brush=pg.mkBrush(COLOR_PHOSPHOR), name="Predicted Target Time")
        self.scatter_triggers = pg.ScatterPlotItem(size=9, symbol='x', pen=pg.mkPen(COLOR_AMBER, width=2), name="Hardware Trigger")
        self.p5.addItem(self.scatter_lookahead)
        self.p5.addItem(self.scatter_triggers)
        layout.addWidget(self.p5, 1, 0, 1, 2)

    def setup_tab_phase(self):
        layout = QGridLayout(self.tab_phase)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        self.p2 = pg.PlotWidget()
        self._style_plot(self.p2, "Phase Estimates Over Time")
        self.p2.addLegend()
        self.curve_active_phase = self.p2.plot(pen=pg.mkPen(COLOR_TEXT_PRIMARY, width=2), name="Active Phase")
        self.curve_kalman_phase = self.p2.plot(pen=pg.mkPen(COLOR_AMBER, width=1.6, style=Qt.DashLine), name="Kalman Phase")
        self.curves_est_phase = {}
        layout.addWidget(self.p2, 0, 0)

        self.p3 = pg.PlotWidget()
        self._style_plot(self.p3, "Kalman Phase Velocity")
        self.curve_k_velocity = self.p3.plot(pen=pg.mkPen(COLOR_BLUE, width=1.6))
        layout.addWidget(self.p3, 0, 1)

        self.p7 = pg.PlotWidget()
        self._style_plot(self.p7, "Delta Phase vs Phase")
        self.p7.setLabel('bottom', 'Phase (radians)')
        self.p7.setLabel('left', 'Delta Phase (radians)')
        self.p7.setXRange(0, 2 * np.pi)
        self.scatter_delta_phase = {}
        layout.addWidget(self.p7, 1, 0, 1, 2)

    def setup_tab_alignment(self):
        layout = QGridLayout(self.tab_alignment)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        self.p1 = pg.PlotWidget()
        self._style_plot(self.p1, "Model Fit / Alignment Scores (Reduced \u03c7\u00b2)")
        self.p1.addLegend()
        self.curves_chi = {}
        layout.addWidget(self.p1, 0, 0)

        self.p6 = pg.PlotWidget()
        self._style_plot(self.p6, "Drift Vector Tracking (pixels)")
        self.p6.addLegend()
        self.curves_drift_x = {}
        self.curves_drift_y = {}
        layout.addWidget(self.p6, 0, 1)

        self.p8 = pg.PlotWidget()
        self._style_plot(self.p8, "System State Classifier Probabilities (Outliers & Best Bin Match)")
        self.p8.addLegend()
        self.curves_p_anon = {}
        self.curves_p_best = {}
        layout.addWidget(self.p8, 1, 0, 1, 2)

    def setup_tab_peak_locking(self):
        layout = QGridLayout(self.tab_peak_locking)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        self.p11 = pg.PlotWidget()
        self._style_plot(self.p11, "Phase Sub-bin Offsets Distribution")
        self.p11.addItem(pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen(COLOR_ROSE, width=1.4, style=Qt.DashLine)))
        self.bg_sub_bin = {}
        layout.addWidget(self.p11, 0, 0)

        self.p12 = pg.PlotWidget()
        self._style_plot(self.p12, "Fractional Spatial Drift Histogram")
        self.p12.addItem(pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen(COLOR_ROSE, width=1.4, style=Qt.DashLine)))
        self.bg_frac_x = {}
        layout.addWidget(self.p12, 0, 1)

        self.p13 = pg.PlotWidget()
        self._style_plot(self.p13, "2D Fractional Pixel Error Scatter")
        self.p13.setXRange(-0.5, 0.5)
        self.p13.setYRange(-0.5, 0.5)
        self.p13.addItem(pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen(COLOR_TEXT_FAINT, width=0.6, style=Qt.DotLine)))
        self.p13.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen(COLOR_TEXT_FAINT, width=0.6, style=Qt.DotLine)))
        self.scatter_2d_drift = {}
        layout.addWidget(self.p13, 1, 0, 1, 2)
        
    def setup_tab_scores(self):
        layout = QGridLayout(self.tab_scores)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        self.p14 = pg.PlotWidget()
        self._style_plot(self.p14, "Instantaneous Loss Landscape (Phase Domain Aligned)")
        self.p14.setLabel('bottom', 'Phase', units='rad')
        self.p14.setXRange(0, 2 * np.pi)
        self.p14.addLegend()
        self.curves_scores = {}
        self.scatter_best_scores = {}
        layout.addWidget(self.p14, 0, 0, 1, 2)

        self.p15 = pg.PlotWidget()
        self._style_plot(self.p15, "Model Uncertainty Tracking (MLE Estimate Radians)")
        self.p15.addLegend()
        self.curves_uncertainty = {}
        layout.addWidget(self.p15, 1, 0)

        self.p16 = pg.PlotWidget()
        self._style_plot(self.p16, "Raw SAD Residual History Tracking")
        self.p16.addLegend()
        self.curves_sad = {}
        layout.addWidget(self.p16, 1, 1)

    def on_tab_changed(self, index):
        self.refresh_ui_plots(force_replot=True)

    def change_residual_source(self, text):
        if self.worker and self.worker.is_alive():
            self.control_queue.put({"type": "SET_RESIDUAL_SOURCE", "value": text})

    def start_acquisition(self):
        while not self.ui_queue.empty():
            try: self.ui_queue.get_nowait()
            except queue.Empty: break
        while not self.bf_queue.empty():
            try: self.bf_queue.get_nowait()
            except queue.Empty: break
        while not self.fl_queue.empty():
            try: self.fl_queue.get_nowait()
            except queue.Empty: break
        while not self.res_queue.empty():
            try: self.res_queue.get_nowait()
            except queue.Empty: break
        while not self.control_queue.empty():
            try: self.control_queue.get_nowait()
            except queue.Empty: break

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_lamp.set_color(COLOR_GREEN_LAMP)

        self._run_start_time = time.time()
        self._trigger_count = 0
        self._last_seen_status = None
        self.lbl_triggers.setText("0")
        self.runtime_timer.start(1000)

        self.active_backend_source = self.combo_source.currentText()

        enabled_ests = []
        for est, chk in self.estimator_checkboxes.items():
            if chk.isChecked():
                enabled_ests.append(est)
        
        if self.active_backend_source not in enabled_ests:
            enabled_ests.append(self.active_backend_source)

        runtime_config = {
            "PHASE_SOURCE": self.active_backend_source,
            "PREDICTION_METHOD": self.combo_predictor.currentText(),
            "SAVE_BF": self.chk_save_bf.isChecked(),
            "SAVE_FL": self.chk_save_fl.isChecked(),
            "ENABLED_ESTIMATORS": enabled_ests,
            "RESIDUAL_SOURCE": self.combo_res_source.currentText(),
            "PARAM_K_MEAS": self.spin_k_meas.value(),
            "PARAM_K_PROC": self.spin_k_proc.value(),
            "PARAM_BOOTSTRAP": self.spin_bootstrap.value()
        }

        self.stop_event.clear()
        self.worker = multiprocessing.Process(
            target=run_hardware_loop,
            args=(self.ui_queue, self.bf_queue, self.fl_queue, self.res_queue, self.control_queue, self.stop_event, runtime_config),
            daemon=True
        )

        self.worker.start()
        self.ui_timer.start(50)  

    def stop_acquisition(self):
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("STOPPING")
        self.status_lamp.set_color(COLOR_AMBER)
        QApplication.processEvents()

        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                self.worker.terminate()
                self.worker.join()
        self.ui_timer.stop()
        self.runtime_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("IDLE")
        self.status_lamp.set_color(COLOR_TEXT_FAINT)

    def _tick_runtime_label(self):
        if self._run_start_time is None:
            return
        elapsed = int(time.time() - self._run_start_time)
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        self.lbl_runtime.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def handle_acquisition_finished(self, storage_path):
        self.ui_timer.stop()
        self.runtime_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("FINISHED")
        self.status_lamp.set_color(COLOR_PHOSPHOR)
        logger.success(f"Acquisition loop executed completely. Trace metrics saved inside: {storage_path}")

    def handle_error(self, error_msg):
        self.ui_timer.stop()
        self.runtime_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("FAULT")
        self.status_lamp.set_color(COLOR_ROSE)
        logger.error(f"Hardware loop tracking halted: {error_msg}")

    def refresh_ui_plots(self, force_replot=False):
        if not self.worker and not force_replot:
            return

        self._bf_level_counter += 1
        relevel_bf = (self._bf_level_counter % 15 == 0)

        bf_frame = None
        for _ in range(4):
            try: bf_frame = self.bf_queue.get_nowait()
            except queue.Empty: break
        if bf_frame is not None:
            self.bf_image_item.setImage(bf_frame.T, autoLevels=relevel_bf)

        fl_frame = None
        for _ in range(4):
            try: fl_frame = self.fl_queue.get_nowait()
            except queue.Empty: break
        if fl_frame is not None:
            self.fl_image_item.setImage(fl_frame.T, autoLevels=True)
            
        res_frame = None
        for _ in range(4):
            try: res_frame = self.res_queue.get_nowait()
            except queue.Empty: break
        if res_frame is not None:
            self.res_image_item.setImage(res_frame.T, autoLevels=False)
            if self.combo_res_source.currentText() == "SAD":
                self.res_image_item.setLevels([-1500.0, 1500.0])
            else:
                self.res_image_item.setLevels([-5.0, 5.0])

        has_new_data = False
        last_packet = None
        max_batches_per_tick = 25

        for _ in range(max_batches_per_tick):
            try:
                batch = self.ui_queue.get_nowait()

                if isinstance(batch, dict) and "status_type" in batch:
                    if batch["status_type"] == "finished":
                        self.handle_acquisition_finished(batch["storage_path"])
                        return
                    elif batch["status_type"] == "error":
                        self.handle_error(batch["message"])
                        return
                    continue

                has_new_data = True
                for data in batch:
                    last_packet = data

                    self.ui_timestamps.append(data["timestamp"])
                    self.ui_framerates.append(data["framerate"])
                    self.ui_active_phases.append(data["active_phase"] if data["active_phase"] is not None else np.nan)
                    self.ui_periods.append(data["est_period"] if data["est_period"] is not None else np.nan)
                    self.ui_lookaheads.append(data["lookahead"] if data["lookahead"] is not None else np.nan)
                    self.ui_k_phases.append(data["k_phase"] if data["k_phase"] is not None else np.nan)
                    self.ui_k_velocities.append(data["k_velocity"] if data["k_velocity"] is not None else np.nan)

                    if data["trigger"] is not None:
                        self.ui_triggers_x.append(data["trigger"][0])
                        self.ui_triggers_y.append(data["trigger"][1])
                        self._trigger_count += 1
                        self.lbl_triggers.setText(f"{self._trigger_count:,}")
                        self.status_lamp.flash(COLOR_AMBER)

                    if not self.est_names and data["estimators"]:
                        self.est_names = list(data["estimators"].keys())
                        for name in self.est_names:
                            self.ui_est_phases[name] = deque(maxlen=self.ui_window)
                            self.ui_chi_squares[name] = deque(maxlen=self.ui_window)
                            self.ui_drift_xs[name] = deque(maxlen=self.ui_window)
                            self.ui_drift_ys[name] = deque(maxlen=self.ui_window)
                            self.ui_p_anons[name] = deque(maxlen=self.ui_window)
                            self.ui_p_bests[name] = deque(maxlen=self.ui_window)
                            
                            self.ui_uncertainties[name] = deque(maxlen=self.ui_window)
                            self.ui_sad_scores[name] = deque(maxlen=self.ui_window)

                            self.raw_vertex_offsets[name] = deque(maxlen=1000)
                            self.raw_frac_xs[name] = deque(maxlen=1000)
                            self.raw_frac_ys[name] = deque(maxlen=1000)
                            self.raw_phases_chunk[name] = deque(maxlen=1000)

                    for name in self.est_names:
                        est = data["estimators"].get(name, {})

                        self.ui_est_phases[name].append(est.get("phase") if est.get("phase") is not None else np.nan)
                        self.ui_chi_squares[name].append(est.get("reduced_chi_squared") if est.get("reduced_chi_squared") is not None else np.nan)
                        self.ui_drift_xs[name].append(est.get("drift_x") if est.get("drift_x") is not None else np.nan)
                        self.ui_drift_ys[name].append(est.get("drift_y") if est.get("drift_y") is not None else np.nan)
                        self.ui_p_anons[name].append(est.get("unknown_anomaly") if est.get("unknown_anomaly") is not None else np.nan)
                        self.ui_p_bests[name].append(est.get("best_bin_prob") if est.get("best_bin_prob") is not None else np.nan)

                        self.ui_uncertainties[name].append(est.get("uncertainty") if est.get("uncertainty") is not None else np.nan)
                        self.ui_sad_scores[name].append(est.get("sad_score") if est.get("sad_score") is not None else np.nan)

                        if est.get("vertex_offset") is not None:
                            self.raw_vertex_offsets[name].append(est["vertex_offset"])
                        if est.get("drift_x") is not None:
                            self.raw_frac_xs[name].append(est["drift_x"] - np.round(est["drift_x"]))
                        if est.get("drift_y") is not None:
                            self.raw_frac_ys[name].append(est["drift_y"] - np.round(est["drift_y"]))
                        if est.get("phase") is not None:
                            self.raw_phases_chunk[name].append(est["phase"])
                            
                        if est.get("scores") is not None:
                            self.latest_scores[name] = est.get("scores")
                            self.latest_best_idx[name] = est.get("best_index")
                            self.latest_v_offset[name] = est.get("vertex_offset")
                            self.latest_ref_period[name] = est.get("reference_period")

            except queue.Empty:
                break

        if not has_new_data and not force_replot:
            return

        if last_packet is not None:
            self.lbl_frame.setText(f"{last_packet['iteration']:,}")
            self.lbl_status.setText(f"{last_packet['status']}")
            self.lbl_fps.setText(f"{last_packet['framerate']:.1f}")

            status_color_map = {"READY": COLOR_PHOSPHOR, "INITIALIZING": COLOR_AMBER}
            new_status = last_packet['status']
            if new_status != self._last_seen_status:
                self._last_seen_status = new_status
                self.status_lamp.set_color(status_color_map.get(new_status, COLOR_GREEN_LAMP))

            active_phase = last_packet.get("active_phase")
            self.phase_dial.set_phase(
                active_phase if active_phase is not None else None,
                target_phase=None,
                armed=last_packet["trigger"] is not None
            )

        t = np.array(self.ui_timestamps)
        if t.size == 0:
            return

        current_tab = self.tabs.currentIndex()

        if current_tab == 0:
            self.curve_framerate.setData(t, np.array(self.ui_framerates))

            y_period = np.array(self.ui_periods)
            if not np.all(np.isnan(y_period)):
                self.curve_period.setData(t, y_period)

            y_lookahead = np.array(self.ui_lookaheads)
            valid_look = ~np.isnan(y_lookahead)
            if np.any(valid_look):
                self.scatter_lookahead.setData(x=t[valid_look], y=t[valid_look] + y_lookahead[valid_look])

            if self.ui_triggers_x:
                tx = np.array(self.ui_triggers_x)
                ty = np.array(self.ui_triggers_y)
                mask = (tx >= t[0]) & (tx <= t[-1])
                if np.any(mask):
                    self.scatter_triggers.setData(x=tx[mask], y=ty[mask])
                else:
                    self.scatter_triggers.setData(x=[], y=[])

        elif current_tab == 1:
            self.curve_active_phase.setData(t, np.array(self.ui_active_phases))

            yk = np.array(self.ui_k_phases)
            if not np.all(np.isnan(yk)):
                self.curve_kalman_phase.setData(t, yk)

            yv = np.array(self.ui_k_velocities)
            if not np.all(np.isnan(yv)):
                self.curve_k_velocity.setData(t, yv)

            for idx, name in enumerate(self.est_names):
                if name not in self.ui_est_phases: continue
                color = SERIES_COLORS[idx % len(SERIES_COLORS)]
                if name not in self.curves_est_phase:
                    self.curves_est_phase[name] = self.p2.plot(pen=pg.mkPen(color, width=1, style=Qt.DotLine), name=f"{name} Estimate")
                self.curves_est_phase[name].setData(t, np.array(self.ui_est_phases[name]))

                p_chunk = np.array(self.raw_phases_chunk[name])
                if p_chunk.size > 1:
                    dp = np.diff(np.unwrap(p_chunk))
                    xp = np.mod(p_chunk[:-1], 2 * np.pi)
                    if name not in self.scatter_delta_phase:
                        self.scatter_delta_phase[name] = pg.ScatterPlotItem(size=4, brush=pg.mkBrush(color))
                        self.p7.addItem(self.scatter_delta_phase[name])
                    self.scatter_delta_phase[name].setData(x=xp, y=dp)

        elif current_tab == 2:
            for idx, name in enumerate(self.est_names):
                if name not in self.ui_chi_squares: continue
                color = SERIES_COLORS[idx % len(SERIES_COLORS)]
                if name not in self.curves_chi:
                    self.curves_chi[name] = self.p1.plot(pen=pg.mkPen(color, width=1.6), name=f"{name} Reduced \u03c7\u00b2")
                self.curves_chi[name].setData(t, np.array(self.ui_chi_squares[name]))

                if name not in self.curves_drift_x:
                    self.curves_drift_x[name] = self.p6.plot(pen=pg.mkPen(color, width=1.6), name=f"{name} Drift X")
                    self.curves_drift_y[name] = self.p6.plot(pen=pg.mkPen(color, width=1.6, style=Qt.DashLine), name=f"{name} Drift Y")
                self.curves_drift_x[name].setData(t, np.array(self.ui_drift_xs[name]))
                self.curves_drift_y[name].setData(t, np.array(self.ui_drift_ys[name]))

                # Decouple shared colors: use distinct dashed styles to link estimator properties together
                if name not in self.curves_p_anon:
                    self.curves_p_anon[name] = self.p8.plot(pen=pg.mkPen(color, width=1.6, style=Qt.DashLine), name=f"{name} Anomaly")
                    self.curves_p_best[name] = self.p8.plot(pen=pg.mkPen(color, width=1.6, style=Qt.SolidLine), name=f"{name} Best Bin Prob")
                self.curves_p_anon[name].setData(t, np.array(self.ui_p_anons[name]))
                self.curves_p_best[name].setData(t, np.array(self.ui_p_bests[name]))

        elif current_tab == 3:
            for idx, name in enumerate(self.est_names):
                if name not in self.raw_vertex_offsets: continue
                color = SERIES_COLORS[idx % len(SERIES_COLORS)]
                v_arr = np.array(self.raw_vertex_offsets[name])
                if v_arr.size > 0:
                    h_y, h_x = np.histogram(v_arr, bins=50, range=(-1.0, 1.0))
                    xc = 0.5 * (h_x[:-1] + h_x[1:])
                    if name not in self.bg_sub_bin:
                        self.bg_sub_bin[name] = pg.BarGraphItem(x=xc, height=h_y, width=0.03, brush=pg.mkBrush(color))
                        self.p11.addItem(self.bg_sub_bin[name])
                    else:
                        self.bg_sub_bin[name].setOpts(x=xc, height=h_y)

                fx = np.array(self.raw_frac_xs[name])
                fy = np.array(self.raw_frac_ys[name])
                if fx.size > 0 and fy.size > 0:
                    h_y_x, h_x_x = np.histogram(fx, bins=50, range=(-0.5, 0.5))
                    xcx = 0.5 * (h_x_x[:-1] + h_x_x[1:])
                    if name not in self.bg_frac_x:
                        self.bg_frac_x[name] = pg.BarGraphItem(x=xcx, height=h_y_x, width=0.01, brush=pg.mkBrush(color))
                        self.p12.addItem(self.bg_frac_x[name])
                    else:
                        self.bg_frac_x[name].setOpts(x=xcx, height=h_y_x)

                    if name not in self.scatter_2d_drift:
                        self.scatter_2d_drift[name] = pg.ScatterPlotItem(size=4, brush=pg.mkBrush(color))
                        self.p13.addItem(self.scatter_2d_drift[name])
                    self.scatter_2d_drift[name].setData(x=fx, y=fy)

        elif current_tab == 4:
            for idx, name in enumerate(self.est_names):
                if name not in self.ui_uncertainties: continue
                color = SERIES_COLORS[idx % len(SERIES_COLORS)]
                
                if name in self.latest_scores and self.latest_scores[name] is not None:
                    scores_arr = np.array(self.latest_scores[name])
                    
                    # Convert raw indices to radians based on each estimator's indexing format
                    if name == "SAD":
                        idx_offset = Config.Gating.SAD_NUM_EXTRA_REF_FRAMES
                        ref_period = self.latest_ref_period.get(name) or (len(scores_arr) - 2 * idx_offset)
                        x_phase = ((np.arange(len(scores_arr)) - idx_offset) / ref_period) * 2 * np.pi
                    else:
                        idx_offset = 0
                        ref_period = len(scores_arr)
                        x_phase = (np.arange(len(scores_arr)) / ref_period) * 2 * np.pi
                    
                    if name not in self.curves_scores:
                        self.curves_scores[name] = self.p14.plot(pen=pg.mkPen(color, width=1.6), name=f"{name} Scores")
                        self.scatter_best_scores[name] = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(color))
                        self.p14.addItem(self.scatter_best_scores[name])
                        
                    self.curves_scores[name].setData(x=x_phase, y=scores_arr)
                    
                    best_idx = self.latest_best_idx.get(name)
                    v_off = self.latest_v_offset.get(name)
                    if best_idx is not None and v_off is not None:
                        exact_x_idx = best_idx + v_off - idx_offset
                        exact_x_phase = (exact_x_idx / ref_period) * 2 * np.pi
                        
                        safe_idx = int(np.clip(best_idx, 0, len(scores_arr) - 1))
                        self.scatter_best_scores[name].setData(x=[exact_x_phase], y=[scores_arr[safe_idx]])
                        
                y_unc = np.array(self.ui_uncertainties[name])
                y_sad = np.array(self.ui_sad_scores[name])
                
                if name not in self.curves_uncertainty:
                    self.curves_uncertainty[name] = self.p15.plot(pen=pg.mkPen(color, width=1.6), name=f"{name} MLE Uncertainty")
                if name not in self.curves_sad:
                    self.curves_sad[name] = self.p16.plot(pen=pg.mkPen(color, width=1.6, style=Qt.DashLine), name=f"{name} SAD Trace")
                    
                if not np.all(np.isnan(y_unc)):
                    self.curves_uncertainty[name].setData(t, y_unc)
                if not np.all(np.isnan(y_sad)):
                    self.curves_sad[name].setData(t, y_sad)

    def closeEvent(self, event):
        self.stop_acquisition()
        event.accept()


if __name__ == "__main__":
    QApplication.setStyle('Fusion')
    multiprocessing.set_start_method('spawn', force=True) if os.name == 'nt' else None
    app = QApplication(sys.argv)
    app.setPalette(build_dark_palette())
    window = LiveDashboardWindow()
    window.show()
    sys.exit(app.exec_())