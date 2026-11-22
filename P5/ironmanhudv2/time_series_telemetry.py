import cv2
import numpy as np

from ironmanhudv2.frame_canvas import FrameCanvas
from ironmanhudv2.rectangle import get_rectangle_draw_points
from ironmanhudv2.units import Point, Size

class TimeSeriesGraph:
    """
    Used to render a time series graph of values over time on the HUD.
    """
    
    def __init__(self, 
                 values: list[float], 
                 normalize_range: tuple[float, float],
                 threshold: float = None
        ):
        self.values = values
        self.normalize_range = normalize_range
        self.threshold = threshold
        
        self._min_divisor = max(self.normalize_range[1] - self.normalize_range[0], 1e-6)
        
    def _draw_graph_bg(self, frame: FrameCanvas, position: Point, size: Size):
        # Background
        pt1, pt2 = get_rectangle_draw_points(position, size)
        cv2.rectangle(
            frame.canvas,
            pt1, pt2,
            (14, 18, 24),  # Dark gray background
            -1
        )
        
        # Border
        cv2.rectangle(
            frame.canvas,
            pt1,
            pt2,
            (55, 70, 82),  # Border color
            1
        )
        
        graph_bg_position = Point(position.x + 1, position.y + 1)
        graph_bg_size = Size(size.width - 2, size.height - 2)
        return graph_bg_position, graph_bg_size
        
    def _get_bar_height(self, value: float, size: Size) -> int:
        clipped_value = np.clip((value - self.normalize_range[0]) / self._min_divisor, 0.0, 1.0)
        return int(size.height * float(clipped_value))
        
    def _draw_threshold_dotted_line(self, frame: FrameCanvas, position: Point, size: Size, dots_spacing: int = 8):
        if self.threshold is None:
            return
        
        bar_height = self._get_bar_height(self.threshold, size)
        threshold_y = position.y + size.height - bar_height
        
        dash_length = max(1, dots_spacing // 2)
        for point_x in range(int(position.x), int(position.x + size.width), dots_spacing):
            cv2.line(
                frame.canvas,
                (point_x, int(threshold_y)),
                (min(point_x + dash_length, int(position.x + size.width - 1)), int(threshold_y)),
                (120, 140, 160),  # Dotted line color
                1
            )
        
    def _calculate_time_points(self, frame: FrameCanvas, position: Point, size: Size, values: list[float]) -> list[Point]:
        time_points = []
        drawable_width = max(size.width - 4, 1)
        drawable_height = max(size.height - 4, 1)

        for i, value in enumerate(values):
            x_offset = int((i / max(len(values) - 1, 1)) * drawable_width)
            x = position.x + 2 + x_offset

            bar_height = self._get_bar_height(value, Size(drawable_width, drawable_height))
            y = position.y + size.height - 2 - bar_height

            time_points.append(Point(x, y))

        return time_points
            
    def _plot_points_over_time(self, frame: FrameCanvas, position: Point, size: Size, values: list[float]):
        if len(self.values) < 2:
            return
        
        points = self._calculate_time_points(frame, position, size, values)
        pts = np.array([p.to_tuple() for p in points], dtype=np.int32)
        cv2.polylines(
            frame.canvas,
            [pts],
            False,
            (0, 255, 0),  # Line color (green in this case)
            1,
            cv2.LINE_AA
        )
        
        
    def render(self, frame: FrameCanvas, position: Point, size: Size, values: list[float]):
        # Render the time series graph on the HUD
        # This could involve drawing lines or bars representing the values over time
        self.values = values
        graph_bg_position, graph_bg_size = self._draw_graph_bg(frame, position, size)
        self._plot_points_over_time(frame, position, size, self.values)
        self._draw_threshold_dotted_line(frame, graph_bg_position, graph_bg_size)
        
        
            
class ThresholdProgressBar:
    """
    Used to render a progress bar with a threshold indicator on the HUD.
    """
    
    def __init__(self, 
                 value: float, 
                 threshold: float, 
                 max_value: float):
        self.value = value
        self.threshold = threshold
        self.max_value = max_value
        
    def _draw_bar_bg(self, frame: FrameCanvas, position: Point, size: Size):
        # Draw the background of the progress bar
        pt1, pt2 = get_rectangle_draw_points(position, size)
        cv2.rectangle(frame.canvas,
                      pt1, pt2,
                      (42, 48, 56),
                      -1)
        
    def _get_bar_width(self, value: float, size: Size) -> int:
        clipped_value = np.clip(value / max(self.max_value, 1e-6), 0.0, 1.0)
        return int(size.width * float(clipped_value))
        
    def _draw_inner_value_bar(self, frame: FrameCanvas, position: Point, size: Size):
        bar_width = self._get_bar_width(self.value, size)
        if bar_width <= 0:
            return
        
        pt1, pt2 = get_rectangle_draw_points(position, Size(bar_width, size.height))
        cv2.rectangle(
            frame.canvas,
            pt1,
            pt2,
            (0, 255, 0),  # Color of the inner value bar (green in this case)
            -1
        )
        
    def _draw_border(self, frame: FrameCanvas, position: Point, size: Size):
        pt1, pt2 = get_rectangle_draw_points(position, size)
        cv2.rectangle(
            frame.canvas,
            pt1,
            pt2,
            (95, 115, 130),  # Color of the border
            1
        )
    
    def _draw_threshold_line(self, frame: FrameCanvas, position: Point, size: Size):
        if self.threshold is None:
            return
        
        threshold_x = position.x + self._get_bar_width(self.threshold, size)
        cv2.line(
            frame.canvas,
            (int(threshold_x), int(position.y)),
            (int(threshold_x), int(position.y + size.height)),
            (255, 0, 0),  # Color of the threshold line (red in this case)
            2,
            cv2.LINE_AA
        )

    def render(self, frame: FrameCanvas, position: Point, size: Size, new_value: float):
        self.value = new_value
        self._draw_bar_bg(frame, position, size)
        self._draw_inner_value_bar(frame, position, size)
        self._draw_border(frame, position, size)
        self._draw_threshold_line(frame, position, size)
        

class TimeSeriesTelemetryHUD:
    """
    Used to render a time series telemetry HUD with multiple graphs for different metrics.
    """
    
    def __init__(self,
                 title: str,
                 label: str,
                 progress_bar: ThresholdProgressBar,
                 time_series_graph: TimeSeriesGraph,
                 additional_graphs_render: list[callable] = []):
        self.title = title
        self.label = label
        self.progress_bar = progress_bar
        self.time_series_graph = time_series_graph
        self.additional_graphs_render = additional_graphs_render
        
    def _render_title(self, frame: FrameCanvas, position: Point):
        cv2.putText(
            frame.canvas,
            self.title,
            position.to_int_tuple(),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * frame.ui_scale,
            (255, 255, 255),  # White color for the title
            2,
            cv2.LINE_AA
        )
    
    def _render_label_with_progress_bar(
        self,
        frame: FrameCanvas,
        position: Point,
        size: Size,
        progress_bar_value: float
    ):
        cv2.putText(
            frame.canvas,
            self.label,
            position.to_int_tuple(),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * frame.ui_scale,
            (255, 255, 255),  # White color for the label
            1,
            cv2.LINE_AA
        )
        
        # Render the progress bar next to the label
        progress_bar_size = Size(150, 15)
        progress_bar_position = Point(position.x + size.width - progress_bar_size.width - 20, position.y - (progress_bar_size.height // 1.2))  # Adjust position as needed
        self.progress_bar.render(frame, progress_bar_position, progress_bar_size, progress_bar_value)
    
    def render(self, 
               frame: FrameCanvas, 
               position: Point, 
               size: Size,
               progress_bar_value: float,
               time_series_values: list[float]
        ):
        initial_pos = position * 1 # create a copy
        
        initial_pos.y += 20 * frame.ui_scale 
        self._render_title(frame, initial_pos)
        initial_pos.y += 30 * frame.ui_scale
        
        self._render_label_with_progress_bar(frame, initial_pos, size, progress_bar_value)
        initial_pos.y += 20 * frame.ui_scale
        
        graph_size = Size(size.width - 20, 80)
        self.time_series_graph.render(frame, initial_pos, graph_size, time_series_values)
        initial_pos.y += graph_size.height + 20 * frame.ui_scale
        
        for render_func in self.additional_graphs_render:
            render_func(frame, initial_pos, size)
            initial_pos.y += 25 * frame.ui_scale