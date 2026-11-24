from dataclasses import dataclass

import cv2
import numpy as np
from typing import Literal

from ironmanhudv2.frame_canvas import FrameCanvas
from ironmanhudv2.hud_theme import ColorScalar, HUDTheme
from ironmanhudv2.math import euclid_dist
from ironmanhudv2.biometry_engine.biometric_state import BiometricState
from ironmanhudv2.biometry_engine.mediapipe_constants import LEFT_IRIS_IDX, RIGHT_IRIS_IDX
from ironmanhudv2.units import Point

@dataclass
class _EyeReticleCircleArtifact:
    radius: int
    offset: list[float]
    phase_offset: float = 1.0  # spin speed multiplier (negative = reverse)
    thickness: int = 1

class FacialMeshHUD:
    def __init__(self):
        self._last_now: float | None = None
        self._phase_deg: float = 0.0
    
    def _draw_eye_reticle(
        self,
        frame: FrameCanvas, 
        center: Point, 
        radius: int, 
        phase_deg: float, 
        color: ColorScalar, 
    ) -> None:
        cx, cy = center.to_int_tuple()
        circles = [
            _EyeReticleCircleArtifact(radius=int(radius * 0.2), offset=[]),
            _EyeReticleCircleArtifact(radius=int(radius * 0.4), offset=[0.0, 180.0], thickness=1, phase_offset=-1.5),
            _EyeReticleCircleArtifact(radius=int(radius * 1), offset=[0.0, 120.0, 240.0], thickness=2, phase_offset=0.8),
            _EyeReticleCircleArtifact(radius=int(radius * 1.2), offset=[0.0], thickness=2, phase_offset=-0.6),
            _EyeReticleCircleArtifact(radius=int(radius * 1.6), offset=[90.0], thickness=2, phase_offset=0.4)
        ]
        
        for circle in circles:
            if circle.radius <= 0:
                continue
            
            if len(circle.offset) == 0:
                cv2.circle(
                    frame.canvas, 
                    (cx, cy), 
                    circle.radius, 
                    color, 
                    circle.thickness, 
                    cv2.LINE_AA
                )
            else:
                phase = (phase_deg * circle.phase_offset) % 360.0
                for offset in circle.offset:
                    cv2.ellipse(
                        frame.canvas, 
                        (cx, cy), 
                        (circle.radius, circle.radius), 
                        phase, 
                        phase + offset, 
                        phase + offset + 76.0, 
                        color, 
                        circle.thickness, 
                        cv2.LINE_AA
                    )
    
    def _draw_ocular_reticles(self, 
                             frame: FrameCanvas, 
                             theme: HUDTheme, 
                             pts: np.ndarray,
                             phase_deg: float = 0.0,
                             eye: Literal['left', 'right', 'both'] = 'both'):
        if pts is None or len(pts) == 0 or len(pts) <= LEFT_IRIS_IDX:
            return
        
        if eye in ['left', 'both']:
            dist_l = euclid_dist(pts[33], pts[133]) * 0.42
            radius_l = max(32, int(dist_l))
            self._draw_eye_reticle(
                frame,
                Point(pts[LEFT_IRIS_IDX][0], pts[LEFT_IRIS_IDX][1]),
                radius_l,
                phase_deg,
                theme.primary,
            )
        
        if eye in ['right', 'both']:
            dist_r = euclid_dist(pts[362], pts[263]) * 0.42
            radius_r = max(32, int(dist_r))
            self._draw_eye_reticle(
                frame,
                Point(pts[RIGHT_IRIS_IDX][0], pts[RIGHT_IRIS_IDX][1]),
                radius_r,
                phase_deg,
                theme.primary,
            )
            
    def render(self, 
               frame: FrameCanvas, 
               theme: HUDTheme, 
               bio_state: BiometricState,
               now: float,
    ):
        if not bio_state.face_detected or bio_state.pts_2d is None or len(bio_state.pts_2d) == 0:
            return
        
        # Faster phase for critical state
        use_alert_phase = bio_state.micro_sleep or bio_state.looking_away
        speed = 95.0 * (2.5 if use_alert_phase else 1.0)
        
        last_now = self._last_now
        dt = 0.0 if last_now is None else max(0.0, now - last_now)
        self._last_now = now
        self._phase_deg = self._phase_deg + speed * dt
        phase_deg = self._phase_deg
        
        # Draw ocular reticles
        self._draw_ocular_reticles(
            frame, 
            theme, 
            bio_state.pts_2d * frame.ui_scale, 
            phase_deg=phase_deg,
            eye='right')