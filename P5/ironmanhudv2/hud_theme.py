import math
from dataclasses import dataclass

from ironmanhudv2.biometry_engine.biometric_state import BiometricState

@dataclass
class ColorScalar:
    b: int
    g: int
    r: int
    
    def __tuple__(self) -> tuple[int, int, int]:
        return (self.b, self.g, self.r)

@dataclass
class HUDTheme:
    primary: ColorScalar
    text_primary: ColorScalar
    secondary: ColorScalar
    accent: ColorScalar
    progress_bar: ColorScalar
    time_series_point: ColorScalar
    radar_point: ColorScalar

DEFAULT_HUD_THEME = HUDTheme(
    (255, 225, 0),
    (255, 232, 60),
    (76, 76, 76),
    (245, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
)

WARNING_HUD_THEME = HUDTheme(
    (0, 180, 255), 
    (80, 205, 255), 
    (0, 120, 200), 
    (120, 240, 255),
    (0, 255, 255), 
    (0, 255, 255), 
    (0, 255, 255),
)

CRITICAL_HUD_THEME = HUDTheme(
    (0, 0, 189), 
    (120, 125, 255), 
    (35, 35, 195), 
    (105, 195, 255),
    (0, 0, 189), 
    (0, 0, 189), 
    (0, 0, 189),
)

def resolve_hud_theme(bio: BiometricState, now_ts: float) -> HUDTheme:
    flash = 0.5 + 0.5 * math.sin(now_ts * 14.0)
    
    if bio.ear.eye_closed and bio.micro_sleep:
        return CRITICAL_HUD_THEME
    
    if bio.looking_away or bio.mar.yawning:
        return WARNING_HUD_THEME
    
    return DEFAULT_HUD_THEME
