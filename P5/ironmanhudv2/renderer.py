from curses.panel import panel

import cv2

from ironmanhudv2.frame_canvas import FrameCanvas
from ironmanhudv2.hud_panel import HUDPanel
from ironmanhudv2.time_series_telemetry import ThresholdProgressBar, TimeSeriesGraph, TimeSeriesTelemetryHUD
from ironmanhudv2.units import Point, Size


class IronManHUDRenderer:
    def __init__(self, resize_frame: tuple[int, int] = None):
        self.resize_frame = Size(*resize_frame) if resize_frame else None
        
        self._ocular_telemetry_hud = None
        self._oral_telemetry_hud = None

    def _render_title_bar(self, frame: FrameCanvas):
        pass
    
    def _blink_label(self, frame: FrameCanvas, position: Point, size: Size):
        MOCK_BLINK_COUNT = 5  # Placeholder for the actual blink count
        MOCK_BLINK_BPM = 12  # Placeholder for the actual blink BPM
        
        cv2.putText(
            frame.canvas,
            f"BLINKS: {MOCK_BLINK_COUNT:.2f} | BPM: {MOCK_BLINK_BPM:.2f}",  # Placeholder for the actual blink count and BPM
            position.to_int_tuple(),  # Position of the label
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 * frame.ui_scale,  # Scale the font size based on UI scale
            (255, 255, 255),  # White color for the label
            1,
            cv2.LINE_AA
        )
    
    def _eye_closure_label(self, frame: FrameCanvas, position: Point, size: Size):
        MOCK_EYE_CLOSED_DURATION = 1.2  # Placeholder for the actual eye closure value
        MOCK_EYE_MICRO_SLEEP_SEC = 2.5  # Placeholder for the actual microsleep duration

        cv2.putText(
            frame.canvas,
            f"EYE CLOSURE: {MOCK_EYE_CLOSED_DURATION:4.2f}s / {MOCK_EYE_MICRO_SLEEP_SEC:.1f}s",  # Placeholder for the actual eye closure value
            position.to_int_tuple(),  # Position of the label
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 * frame.ui_scale,  # Scale the font size based on UI scale
            (255, 255, 255),  # White color for the label
            1,
            cv2.LINE_AA
        )
        
    def _format_data_label_callable(self, func: callable):
        return lambda frame, pos, size: func(frame, pos, size)
    
    def _prepare_ocular_telemetry(self):
        # Draw section title "Ocular Telemetry"
        # Calculate EAR (Eye Aspect Ratio)
        # Draw a label "EAR: {value_EAR:.2f}" on the HUD
        # Draw a progress bar next to label with indication threshold
        # Draw below the label and progress bar a time series graph of EAR values over time
        MOCK_VALUE = 0.142  # Placeholder for the actual EAR value
        MOCK_THRESHOLD = 0.3  # Placeholder for the actual threshold value
        MOCK_MAX_VALUE = 0.8  # Placeholder for the actual max value for normalization
        
        progress_bar = ThresholdProgressBar(
            value=MOCK_VALUE, # PLACEHOLDER
            threshold=MOCK_THRESHOLD, # PLACEHOLDER
            max_value=MOCK_MAX_VALUE, # PLACEHOLDER
        )
        
        def random_data_series(num_points: int) -> list[float]:
            import random
            return [random.uniform(0.0, MOCK_MAX_VALUE) for _ in range(num_points)]
        
        time_series_graph = TimeSeriesGraph(
            values=random_data_series(256), # PLACEHOLDER
            normalize_range=(0.0, MOCK_MAX_VALUE), # PLACEHOLDER
            threshold=MOCK_THRESHOLD, # PLACEHOLDER
        )
        
        additional_data_labels = [
            self._format_data_label_callable(self._blink_label),
            self._format_data_label_callable(self._eye_closure_label)
        ]
        
        ocular_telemetry_hud = TimeSeriesTelemetryHUD(
            title="OCULAR TELEMETRY",
            label=f"EAR: {MOCK_VALUE:.2f}", # PLACEHOLDER
            progress_bar=progress_bar,
            time_series_graph=time_series_graph,
            additional_graphs_render=additional_data_labels
        )
        
        return ocular_telemetry_hud        
    
    def _yawn_counter_label(self, frame: FrameCanvas, position: Point, size: Size):
        MOCK_YAWN_COUNT = 3  # Placeholder for the actual yawn count
        MOCK_YAWN_STATE = "NORMAL"
        
        cv2.putText(
            frame.canvas,
            f"YAWNS: {MOCK_YAWN_COUNT} [{MOCK_YAWN_STATE}]",  # Placeholder for the actual yawn count
            position.to_int_tuple(),  # Position of the label
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 * frame.ui_scale,  # Scale the font size based on UI scale
            (255, 255, 255),  # White color for the label
            1,
            cv2.LINE_AA
        )
    
    def _prepare_oral_telemetry(self):
        # Draw section title "Oral Telemetry"
        # Calculate MAR (Mouth Aspect Ratio)
        # Draw a label "MAR: {value_MAR:.2f}" on the HUD
        # Draw a progress bar next to label with indication threshold
        # Draw below the label and progress bar a time series graph of MAR values over time
        MOCK_VALUE = 0.093  # Placeholder for the actual EAR value
        MOCK_THRESHOLD = 0.3  # Placeholder for the actual threshold value
        MOCK_MAX_VALUE = 1.0  # Placeholder for the actual max value for normalization
        
        progress_bar = ThresholdProgressBar(
            value=MOCK_VALUE, # PLACEHOLDER
            threshold=MOCK_THRESHOLD, # PLACEHOLDER
            max_value=MOCK_MAX_VALUE, # PLACEHOLDER
        )
        
        def random_data_series(num_points: int) -> list[float]:
            import random
            return [random.uniform(0.0, MOCK_MAX_VALUE) for _ in range(num_points)]
        
        time_series_graph = TimeSeriesGraph(
            values=random_data_series(256), # PLACEHOLDER
            normalize_range=(0.0, MOCK_MAX_VALUE), # PLACEHOLDER
            threshold=MOCK_THRESHOLD, # PLACEHOLDER
        )
        
        additional_data_labels = [
            self._format_data_label_callable(self._yawn_counter_label),
        ]
        
        ocular_telemetry_hud = TimeSeriesTelemetryHUD(
            title="ORAL / FATIGUE TELEMETRY",
            label=f"MAR: {MOCK_VALUE:.2f}", # PLACEHOLDER
            progress_bar=progress_bar,
            time_series_graph=time_series_graph,
            additional_graphs_render=additional_data_labels
        )
        
        return ocular_telemetry_hud  
    
    def setup(self):
        self._ocular_telemetry_hud = self._prepare_ocular_telemetry()
        self._oral_telemetry_hud = self._prepare_oral_telemetry()
        
    def _is_setup_complete(self) -> bool:
        if self._ocular_telemetry_hud is None:
            return False
        
        if self._oral_telemetry_hud is None:
            return False
    
        return True

    def _render_ocular_oral_telemetry(self, frame: FrameCanvas):
        panel_pos = Point(10, 50).scale(frame.ui_scale)
        panel_size = Size(300, 410).scale(frame.ui_scale)

        panel = HUDPanel(panel_pos, panel_size)        
        panel.render(frame)
        
        pos_offset = Point(10, 10).scale(frame.ui_scale)
        initial_pos = Point(panel_pos.x + pos_offset.x, panel_pos.y + pos_offset.y)
        size_offset = Size(20, 40).scale(frame.ui_scale)
        size = Size(panel_size.width - size_offset.width, (panel_size.height - size_offset.height) // 2) # Size for the time series graph and progress bar
        
        self._ocular_telemetry_hud.render(frame, initial_pos, size)
        
        initial_pos.y += size.height + (10 * (frame.ui_scale * 2))  # Move down for the next graph
        self._oral_telemetry_hud.render(frame, initial_pos, size)

    def render(self, frame: cv2.Mat):
        if not self._is_setup_complete():
            raise RuntimeError("HUD setup is not complete. Please call setup_hud() before rendering.")
        
        frame_canvas = FrameCanvas(frame, resize_frame=self.resize_frame)
        
        #self._render_title_bar()
        self._render_ocular_oral_telemetry(frame_canvas)
        
        return frame_canvas.canvas
        