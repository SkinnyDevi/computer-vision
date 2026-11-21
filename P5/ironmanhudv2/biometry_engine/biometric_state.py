import numpy as np
from dataclasses import dataclass, field

from ironmanhudv2.math import BBox

@dataclass
class ComputedEAR:
    """
    Computed Eye to Ear Ratio.
    Used for detecting eye closure and blinking.
    """
    
    left_eye: float = 0.0
    right_eye: float = 0.0
    value: float = 0.0
    raw: float = 0.0
    eye_closed: bool = False
    eye_closed_duration: float = 0.0
    blink_count: int = 0
    blink_rate_bpm: float = 0.0
    history: list[float] = field(default_factory=list)

@dataclass
class ComputedMAR:
    """
    Compute Mouth to Ear Ratio.
    Used for detecting mouth opening and yawning.
    """
    
    value: float = 0.0
    raw: float = 0.0
    mouth_open: bool = False
    mouth_open_duration: float = 0.0
    yawn_count: int = 0
    yawning: bool = False
    history: list[float] = field(default_factory=list)
    
@dataclass
class Pitch3D:
    """
    3D Pitch (rotation around the X-axis).
    Used for detecting head nodding and tilting.
    """
    
    raw: float = 0.0
    value: float = 0.0

@dataclass
class Yaw3D:
    """
    3D Yaw (rotation around the Y-axis).
    Used for detecting head turning left and right.
    """
    
    raw: float = 0.0
    value: float = 0.0

@dataclass
class Roll3D:
    """
    3D Roll (rotation around the Z-axis).
    Used for detecting head tilting.
    """
    
    raw: float = 0.0
    value: float = 0.0

@dataclass
class BiometricState:
    timestamp: float = 0.0
    face_detected: bool = False
    pts_2d: np.ndarray | None = None
    bbox: BBox = field(default_factory=lambda: BBox(0, 0, 0, 0))
    
    ear: ComputedEAR = field(default_factory=ComputedEAR)
    mar: ComputedMAR = field(default_factory=ComputedMAR)
    
    pitch: Pitch3D = field(default_factory=Pitch3D)
    yaw: Yaw3D = field(default_factory=Yaw3D)
    roll: Roll3D = field(default_factory=Roll3D)
    
    rvec: np.ndarray | None = None # Rotation vector for head pose estimation
    tvec: np.ndarray | None = None # Translation vector for head pose estimation
    dist_coeffs: np.ndarray | None = None # Distortion coefficients for camera calibration
    cam_matrix: np.ndarray | None = None # Camera matrix for head pose estimation
    
    nodding_off: bool = False
    nod_duration: float = 0.0
    looking_away: bool = False
    looking_away_duration: float = 0.0
    