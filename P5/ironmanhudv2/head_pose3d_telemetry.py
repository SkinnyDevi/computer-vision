import cv2

from ironmanhudv2.biometry_engine.biometric_state import BiometricState
from ironmanhudv2.frame_canvas import FrameCanvas
from ironmanhudv2.units import Point, Size


class HeadPose3DTelemetry:
    def __init__(self):
        pass

    def _get_sign_str(self, value: float) -> str:
        return "+" if value >= 0 else ""
        
    def _draw_pitch_value(self, frame: FrameCanvas, pitch: float, position: Point, size: Size):
        pitch_text = f"Pitch: {self._get_sign_str(pitch)}{pitch:.2f} deg"
        cv2.putText(
            frame.canvas,
            pitch_text,
            (int(position.x), int(position.y + size.height // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * frame.ui_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    def _draw_yaw_value(self, frame: FrameCanvas, yaw: float, position: Point, size: Size):
        yaw_text = f"Yaw: {self._get_sign_str(yaw)}{yaw:.2f} deg"
        cv2.putText(
            frame.canvas,
            yaw_text,
            (int(position.x), int(position.y + size.height // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * frame.ui_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    def _draw_roll_value(self, frame: FrameCanvas, roll: float, position: Point, size: Size):
        roll_text = f"Roll: {self._get_sign_str(roll)}{roll:.2f} deg"
        cv2.putText(
            frame.canvas,
            roll_text,
            (int(position.x), int(position.y + size.height // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * frame.ui_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
    def _draw_radar_background(self, frame: FrameCanvas, position: Point, size: Size):
        # Background
        cv2.circle(
            frame.canvas, 
            (int(position.x + size.width // 2), 
             int(position.y + size.height // 2)), 
            int(size.width // 2), 
            (30, 38, 46), 
            -1, 
            cv2.LINE_AA
        )
        
        # Draw the inner radar circle
        cv2.circle(
            frame.canvas,
            (int(position.x + size.width // 2), 
             int(position.y + size.height // 2)),
            int(size.width // 2 // 2),
            (75, 95, 110),
            1,
            cv2.LINE_AA
        )
        
        # Border
        cv2.circle(
            frame.canvas,
            (int(position.x + size.width // 2), 
             int(position.y + size.height // 2)),
            int(size.width // 2),
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    def _draw_radar_axis_lines(self, frame: FrameCanvas, position: Point, size: Size):
        # X axis line
        axis_line_color = (79, 101, 117)
        cv2.line(
            frame.canvas,
            (int(position.x), int(position.y + size.height // 2)),
            (int(position.x + size.width), int(position.y + size.height // 2)),
            axis_line_color,
            1
        )
        # Y axis line
        cv2.line(
            frame.canvas,
            (int(position.x + size.width // 2), int(position.y)),
            (int(position.x + size.width // 2), int(position.y + size.height)),
            axis_line_color,
            1
        )
            
    def _calculate_radar_point(self, state: BiometricState, position: Point, size: Size, pos_multipler: float = 1.0) -> Point:
        # Calculate the radar point based on pitch and yaw
        radar_radius = size.width // 2 // 2  # Inner radar circle radius
        pitch_rad = state.pitch.value * (3.14159 / 180)  # Convert to radians
        yaw_rad = state.yaw.value * (3.14159 / 180)  # Convert to radians
        
        x_offset = int(radar_radius * yaw_rad) * pos_multipler
        y_offset = int(radar_radius * pitch_rad) * pos_multipler
        
        radar_center_x = int(position.x + size.width // 2)
        radar_center_y = int(position.y + size.height // 2)
        
        radar_point_x = radar_center_x + x_offset
        radar_point_y = radar_center_y - y_offset  # Invert y-axis for display
        
        return Point(radar_point_x, radar_point_y)
    
    def _draw_pose_radar_point(self, frame: FrameCanvas, radar_point: Point):
        # Center circle
        cv2.circle(
            frame.canvas,
            (int(radar_point.x), int(radar_point.y)),
            5,
            (0, 255, 0),
            -1,
            cv2.LINE_AA
        )
        
        # Outer circle
        cv2.circle(
            frame.canvas,
            (int(radar_point.x), int(radar_point.y)),
            8,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
    
    def _draw_3d_radar(self, frame: FrameCanvas, state: BiometricState, position: Point, size: Size):
        self._draw_radar_background(frame, position, size)
        self._draw_radar_axis_lines(frame, position, size)
        # Radar point for head position
        radar_pos = self._calculate_radar_point(state, position, size, pos_multipler=1.3)
        self._draw_pose_radar_point(frame, radar_pos)
            
    def render(self,
               frame: FrameCanvas,
               state: BiometricState,
               position: Point, 
               size: Size):
        initial_pos = position * 1 # create a copy
        
        initial_pos.y -= 40 * frame.ui_scale
        self._draw_pitch_value(frame, state.pitch.value, initial_pos, size)
        
        initial_pos.y += 30 * frame.ui_scale
        self._draw_yaw_value(frame, state.yaw.value, initial_pos, size)
        
        initial_pos.y += 30 * frame.ui_scale
        self._draw_roll_value(frame, state.roll.value, initial_pos, size)
        
        original_position = position * 1
        radar_3d_position = Point(original_position.x + size.width - 110, original_position.y + 20)
        radar_3d_size = Size(90, 90)
        self._draw_3d_radar(frame, state, radar_3d_position, radar_3d_size)