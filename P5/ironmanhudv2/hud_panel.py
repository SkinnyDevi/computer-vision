import cv2

from ironmanhudv2.frame_canvas import FrameCanvas
from ironmanhudv2.rectangle import get_rectangle_draw_points
from ironmanhudv2.units import Point, Size


class HUDPanel:
    def __init__(self, 
                 position: Point,
                 size: Size,
        ):
        self.position = position
        self.size = size
        
    def _add_transparent_bg(self, frame: FrameCanvas, alpha: float = 0.74):
        overlay = frame.canvas_copy()
        p1, p2 = get_rectangle_draw_points(self.position, self.size)
        cv2.rectangle(overlay, p1, p2, (12, 16, 22), -1)  # Dark gray background with transparency
        cv2.addWeighted(overlay, alpha, frame.canvas, 1.0 - alpha, 0.0, frame.canvas)

    def _draw_borders(self, frame: FrameCanvas, accent_color: tuple[int, int, int] = (95, 115, 130)):
        p1, p2 = get_rectangle_draw_points(self.position, self.size)
        cv2.rectangle(frame.canvas,
                      p1, p2,
                      accent_color,  # Accent color for the border
                      1,
                      cv2.LINE_AA)
        
    def _draw_rounded_corners(self, frame: FrameCanvas, corner_radius: int = 10, accent_color: tuple[int, int, int] = (95, 115, 130)):
        p1, p2 = get_rectangle_draw_points(self.position, self.size)
        corner_radius = int(min(max(corner_radius, int(min(self.size.width, self.size.height) * 0.08)), self.size.width // 2, self.size.height // 2))  # Ensure a minimum corner radius
        corner_points = (
            (p1[0], p1[1], 1, 1), 
            (p2[0], p1[1], -1, 1), 
            (p1[0], p2[1], 1, -1), 
            (p2[0], p2[1], -1, -1)
        )
        
        for cx, cy, dx, dy in corner_points:
            cv2.line(frame.canvas, (cx, cy), (cx + dx * corner_radius, cy), accent_color, 2, cv2.LINE_AA)
            cv2.line(frame.canvas, (cx, cy), (cx, cy + dy * corner_radius), accent_color, 2, cv2.LINE_AA)

    def render(self, frame: FrameCanvas):
        self._add_transparent_bg(frame)
        self._draw_borders(frame)
        self._draw_rounded_corners(frame)
        