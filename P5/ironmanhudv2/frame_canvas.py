import cv2
import numpy as np

from ironmanhudv2.units import Size

class FrameCanvas:
    """
    Used to render a frame _canvas on the HUD.
    """
    
    def __init__(self, frame: cv2.Mat, resize_frame: Size = None):
        h, w = frame.shape[:2]
        self._width = w
        self._height = h
        self._aspect_ratio = w / h
        self._ui_scale = 1.0
        
        self._canvas = frame.copy()
        
        if resize_frame is not None:
            self._width, self._height = resize_frame.to_tuple()
            self._canvas = cv2.resize(self._canvas, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
            self._aspect_ratio = self._width / self._height
            
            # Adjust the UI scale based on the new height to maintain consistent sizing of HUD elements
            self._ui_scale = float(np.clip(min(self._height / 1024.0, self._height / 720.0), 0.75, 1.30))
        
    @property
    def canvas(self) -> cv2.Mat:
        return self._canvas
    
    def canvas_copy(self) -> cv2.Mat:
        return self._canvas.copy()
    
    @property
    def width(self) -> int:
        return self._width
    
    @property
    def height(self) -> int:
        return self._height
    
    @property
    def size(self) -> Size:
        return Size(self._width, self._height)
    
    @property
    def aspect_ratio(self) -> float:
        return self._aspect_ratio
    
    @property
    def ui_scale(self) -> float:
        return self._ui_scale