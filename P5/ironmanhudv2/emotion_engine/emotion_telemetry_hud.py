import cv2
import numpy as np

from ironmanhudv2.emotion_engine.emotion_engine import EmotionPrediction
from ironmanhudv2.frame_canvas import FrameCanvas
from ironmanhudv2.hud_panel import HUDPanel
from ironmanhudv2.hud_theme import DEFAULT_HUD_THEME, HUDTheme
from ironmanhudv2.time_series_telemetry import ThresholdProgressBar
from ironmanhudv2.units import Point, Size


class EmotionTelemetryHUD:
    """Renders per-emotion confidence progress bars inside a HUD panel."""

    def __init__(
        self,
        title: str = "EMOTION TELEMETRY",
        threshold: float | None = None,
        bar_height: int = 12,
        row_spacing: int = 24,
        sort_by_confidence: bool = False,
        max_emotions: int | None = None,
    ):
        self.title = title
        self.threshold = threshold
        self.bar_height = bar_height
        self.row_spacing = row_spacing
        self.sort_by_confidence = sort_by_confidence
        self.max_emotions = max_emotions
        self._progress_bars: dict[str, ThresholdProgressBar] = {}

    def _get_progress_bar(self, emotion: str) -> ThresholdProgressBar:
        bar = self._progress_bars.get(emotion)
        if bar is None:
            bar = ThresholdProgressBar(value=0.0, threshold=self.threshold, max_value=1.0)
            self._progress_bars[emotion] = bar
        return bar

    def _prepare_emotions(self, emotions: list[EmotionPrediction] | None) -> list[EmotionPrediction]:
        if not emotions:
            return []
        
        prepared = list(emotions) # Create a copy
        if self.sort_by_confidence:
            prepared.sort(key=lambda pred: pred.confidence, reverse=True)
            
        if self.max_emotions is not None:
            prepared = prepared[: self.max_emotions]
            
        return prepared

    def _compute_default_panel_size(self, frame: FrameCanvas, num_rows: int) -> Size:
        s = frame.ui_scale
        width = 300.0 * s
        height = 68.0 * s if num_rows == 0 else (76.0 + num_rows * self.row_spacing + 12.0) * s
        return Size(width, height)

    def _render_title(self, frame: FrameCanvas, position: Point) -> None:
        cv2.putText(
            frame.canvas,
            self.title,
            position.to_int_tuple(),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * frame.ui_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _render_emotion_row(
        self,
        frame: FrameCanvas,
        position: Point,
        content_width: float,
        prediction: EmotionPrediction,
        theme: HUDTheme,
        is_dominant: bool,
    ) -> None:
        confidence = float(np.clip(prediction.confidence, 0.0, 1.0))
        s = frame.ui_scale
        label_col_w = 88.0 * s
        pct_col_w = 40.0 * s
        gap = 6.0 * s

        label_color = theme.accent if is_dominant else (220, 232, 240)
        pct_color = theme.text_primary if is_dominant else (220, 232, 240)

        cv2.putText(
            frame.canvas,
            prediction.emotion.upper(),
            position.to_int_tuple(),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42 * s,
            label_color,
            1,
            cv2.LINE_AA,
        )

        bar_w = max(40.0 * s, content_width - label_col_w - pct_col_w - gap)
        bar_h = max(6.0, self.bar_height * s)
        bar_pos = Point(position.x + label_col_w, position.y - bar_h * 0.85)
        self._get_progress_bar(prediction.emotion).render(
            frame,
            bar_pos,
            Size(bar_w, bar_h),
            confidence,
            theme,
        )

        cv2.putText(
            frame.canvas,
            f"{confidence * 100:3.0f}%",
            Point(bar_pos.x + bar_w + gap, position.y).to_int_tuple(),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40 * s,
            pct_color,
            1,
            cv2.LINE_AA,
        )

    def _render_hud_panel(
        self,
        frame: FrameCanvas,
        emotions: list[EmotionPrediction] | None,
        theme: HUDTheme = DEFAULT_HUD_THEME,
        position: Point | None = None,
        panel: HUDPanel | None = None,
        size: Size | None = None,
        draw_panel: bool = True,
    ) -> None:
        if position is None and panel is None:
            raise ValueError("Either position or panel must be provided.")

        display_emotions = self._prepare_emotions(emotions)
        s = frame.ui_scale

        if panel is None:
            origin = position * 1
            area_size = size if size is not None else self._compute_default_panel_size(frame, len(display_emotions))
            if draw_panel:
                HUDPanel(position=origin, size=area_size).render(frame, theme)
        else:
            if panel._hud_theme is None:
                panel.render(frame, theme)
            origin = (position * 1) if position is not None else (panel.position * 1)
            area_size = (
                size
                if size is not None
                else Size(
                    max(80.0, (panel.position.x + panel.size.width) - origin.x),
                    max(40.0, (panel.position.y + panel.size.height) - origin.y),
                )
            )

        content_x = origin.x + 12.0 * s
        content_w = max(80.0, area_size.width - 24.0 * s)
        max_y = origin.y + area_size.height - 6.0 * s
        cursor_y = origin.y + 24.0 * s

        self._render_title(frame, Point(content_x, cursor_y))
        cursor_y += 22.0 * s

        if not display_emotions:
            cv2.putText(
                frame.canvas,
                "NO FACE DETECTED",
                Point(content_x, cursor_y).to_int_tuple(),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45 * s,
                (160, 175, 190),
                1,
                cv2.LINE_AA,
            )
            return

        dominant = max(emotions, key=lambda pred: pred.confidence)
        cv2.putText(
            frame.canvas,
            f"DOMINANT: {dominant.emotion.upper()} ({dominant.confidence * 100:.0f}%)",
            Point(content_x, cursor_y).to_int_tuple(),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 * s,
            theme.text_primary,
            1,
            cv2.LINE_AA,
        )

        cursor_y += 8.0 * s
        cv2.line(
            frame.canvas,
            Point(content_x, cursor_y).to_int_tuple(),
            Point(content_x + content_w, cursor_y).to_int_tuple(),
            (60, 80, 95),
            1,
        )

        cursor_y += 22.0 * s
        for prediction in display_emotions:
            if cursor_y > max_y:
                break
            self._render_emotion_row(
                frame,
                Point(content_x, cursor_y),
                content_w,
                prediction,
                theme,
                is_dominant=(prediction.emotion == dominant.emotion),
            )
            cursor_y += self.row_spacing * s

    def render(
        self,
        frame: FrameCanvas,
        position: Point,
        emotions: list[EmotionPrediction] | None,
        theme: HUDTheme = DEFAULT_HUD_THEME,
        size: Size | None = None,
        draw_panel: bool = True,
    ) -> None:
        self._render_hud_panel(
            frame,
            emotions=emotions,
            theme=theme,
            position=position,
            size=size,
            draw_panel=draw_panel,
        )

    def render_in(
        self,
        frame: FrameCanvas,
        panel: HUDPanel,
        emotions: list[EmotionPrediction] | None,
        theme: HUDTheme = DEFAULT_HUD_THEME,
        position: Point | None = None,
        size: Size | None = None,
    ) -> None:
        self._render_hud_panel(
            frame,
            emotions=emotions,
            theme=theme,
            position=position,
            panel=panel,
            size=size,
        )
