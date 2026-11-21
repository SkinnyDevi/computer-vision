import numpy as np

from ironmanhudv2.math import euclid_dist
from ironmanhudv2.biometry_engine.mediapipe_constants import MOUTH_CORNERS, MOUTH_VERT_PAIRS

def compute_EAR(pts_2d: np.ndarray, eye_indices: list[int]) -> float:
    """
    Compute Eye to Ear Ratio:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
    where:
        p1, p2, p3, p4, p5, p6 are the 2D coordinates of the eye landmarks.
        
    Used for detecting eye closure and blinking.
    """
    
    p1, p2, p3, p4, p5, p6 = [pts_2d[i] for i in eye_indices]
    horiz = euclid_dist(p1, p4)
    return 0.0 if horiz < 1e-6 else (euclid_dist(p2, p6) + euclid_dist(p3, p5)) / (2.0 * horiz)


def compute_MAR(pts_2d: np.ndarray) -> float:
    """
    Compute Mouth to Ear Ratio:
    MAR = (||p2 - p6|| + ||p3 - p5||) / ||p1 - p4||
    
    where:
        p1, p2, p3, p4, p5, p6 are the 2D coordinates of the mouth landmarks.
        
    Used for detecting mouth opening and yawning.
    """
    
    horiz = euclid_dist(pts_2d[MOUTH_CORNERS[0]], pts_2d[MOUTH_CORNERS[1]])
    if horiz < 1e-6:
        return 0.0
    
    vert_avg = sum(euclid_dist(pts_2d[a], pts_2d[b]) for a, b in MOUTH_VERT_PAIRS) / len(MOUTH_VERT_PAIRS)
    return vert_avg / horiz