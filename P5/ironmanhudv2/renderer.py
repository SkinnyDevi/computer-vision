from pathlib import Path

import cv2
import time
import mediapipe as mp
import torch

from ironmanhudv2.alarm_audio import AudioAlarm
from ironmanhudv2.biometry_engine.biometric_state import BiometricState
from ironmanhudv2.biometry_engine.tracker import BiometricTracker
from ironmanhudv2.emotion_engine.emotion_engine import EmotionEngine
from ironmanhudv2.biometry_engine.engine_config import BiometryConfig

from ironmanhudv2.emotion_engine.emotion_telemetry_hud import EmotionTelemetryHUD
from ironmanhudv2.facial_mesh import FacialMeshHUD
from ironmanhudv2.frame_canvas import FrameCanvas
from ironmanhudv2.head_pose3d_telemetry import HeadPose3DTelemetry
from ironmanhudv2.hud_panel import HUDPanel
from ironmanhudv2.hud_theme import resolve_hud_theme, HUDTheme
from ironmanhudv2.time_series_telemetry import ThresholdProgressBar, TimeSeriesGraph, TimeSeriesTelemetryHUD
from ironmanhudv2.units import Point, Size

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

class IronManHUDRenderer:
    def __enter__(self):
            return self
    
    def __exit__(self):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
    
    def __init__(self, resize_frame: tuple[int, int] = None):
        self.resize_frame = Size(*resize_frame) if resize_frame else None
        
        self.prev_frame_time = None
        self._last_ts_ms: int = 0
        
        self._biometry_config: BiometryConfig | None = None
        self._biometric_tracker: BiometricTracker | None = None
        self._landmarker = None
        self._ocular_telemetry_hud = None
        self._oral_telemetry_hud = None
        self._head_pose_telemetry = None
        self._facial_mesh = None
        self._emotion_telemetry_hud = None
        
        self._audio_alarm = None
        self._emotion_engine: EmotionEngine | None = None
        self._bio_state: BiometricState = BiometricState()
        self._current_hud_theme: HUDTheme = resolve_hud_theme(self._bio_state, 0.0)

    def _render_title_bar(self, frame: FrameCanvas, fps: float):
        panel_pos = Point(10, 10).scale(frame.ui_scale)
        panel_size = Size(frame.width - 20, 30).scale(frame.ui_scale)
        
        panel = HUDPanel(panel_pos, panel_size)
        panel.render(frame, self._current_hud_theme)
        
        # Title
        text_offset = Point(15, 22).scale(frame.ui_scale)
        panel_text_pos = panel_pos + text_offset
        cv2.putText(
            frame.canvas,
            "IRONMAN HUD V2",
            panel_text_pos.to_int_tuple(),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * frame.ui_scale,
            (255, 255, 255),
            2,
        )
        
        # Other Metrics
        other_metrics_offset = Point(panel_size.width - 50, 29).scale(frame.ui_scale)
        cv2.putText(
            frame.canvas,
            f"FPS: {fps:.0f}",
            other_metrics_offset.to_int_tuple(),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4 * frame.ui_scale,
            (255, 255, 0),
            1,
        )

    def _blink_label(self, frame: FrameCanvas, position: Point, size: Size):
        blink_count = self._bio_state.ear.blink_count
        blink_bpm = self._bio_state.ear.blink_rate_bpm
        cv2.putText(
            frame.canvas,
            f"BLINKS: {blink_count} | BPM: {blink_bpm:.2f}",
            position.to_int_tuple(),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 * frame.ui_scale,
            (255, 255, 255),  # White color for the label
            1,
            cv2.LINE_AA
        )
    
    def _eye_closure_label(self, frame: FrameCanvas, position: Point, size: Size):
        closed_duration = self._bio_state.ear.eye_closed_duration
        cv2.putText(
            frame.canvas,
            f"EYE CLOSURE: {closed_duration:4.2f}s / {self._biometry_config.params_EAR.micro_sleep_seconds:.1f}s",
            position.to_int_tuple(),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 * frame.ui_scale,
            (255, 255, 255),  # White color for the label
            1,
            cv2.LINE_AA
        )
        
    def _format_data_label_callable(self, func: callable):
        return lambda frame, pos, size: func(frame, pos, size)
    
    def _prepare_ocular_telemetry(self):
        progress_bar = ThresholdProgressBar(
            value=0,
            threshold=self._biometry_config.params_EAR.threshold,
            max_value=0.42,
        )
        
        time_series_graph = TimeSeriesGraph(
            values=[],
            normalize_range=(0.05, 0.42),
            threshold=self._biometry_config.params_EAR.threshold,
        )
        
        additional_data_labels = [
            self._format_data_label_callable(self._blink_label),
            self._format_data_label_callable(self._eye_closure_label)
        ]
        
        return TimeSeriesTelemetryHUD(
            title="OCULAR TELEMETRY",
            label=f"EAR: {0:.2f}",
            progress_bar=progress_bar,
            time_series_graph=time_series_graph,
            additional_graphs_render=additional_data_labels
        )
    
    def _yawn_counter_label(self, frame: FrameCanvas, position: Point, size: Size):
        MOCK_YAWN_STATE = "NORMAL"
        
        yawn_count = self._bio_state.mar.yawn_count
        cv2.putText(
            frame.canvas,
            f"YAWNS: {yawn_count} [{MOCK_YAWN_STATE}]",
            position.to_int_tuple(),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 * frame.ui_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    def _prepare_oral_telemetry(self):
        progress_bar = ThresholdProgressBar(
            value=0,
            threshold=self._biometry_config.params_MAR.threshold,
            max_value=0.90,
        )
        
        time_series_graph = TimeSeriesGraph(
            values=[],
            normalize_range=(0.0, 0.90),
            threshold=self._biometry_config.params_MAR.threshold,
        )
        
        additional_data_labels = [
            self._format_data_label_callable(self._yawn_counter_label),
        ]
        
        return TimeSeriesTelemetryHUD(
            title="ORAL / FATIGUE TELEMETRY",
            label=f"MAR: {0:.2f}",
            progress_bar=progress_bar,
            time_series_graph=time_series_graph,
            additional_graphs_render=additional_data_labels
        )
    
    def _prepare_head_pose_telemetry(self):
        return HeadPose3DTelemetry()
    
    def _prepare_facial_mesh(self):
        return FacialMeshHUD()
    
    def _prepare_emotion_engine(self, bio: BiometryConfig):
        return EmotionEngine(torch_model=bio.emotion_model_path)
    
    def _prepare_emotion_telemetry(self, bio: BiometryConfig):
        return EmotionTelemetryHUD(
            title="EMOTION TELEMETRY",
            threshold=bio.emotion_conf_threshold,
            bar_height=12,
            row_spacing=24,
            sort_by_confidence=True,
            max_emotions=7
        )
    
    def _prepare_audio_alarm(self, bio: BiometryConfig):
        return AudioAlarm(
            wav_path=bio.alarm_wav_path,
            cooldown_seconds=bio.alarm_cooldown_seconds
        )

    def setup(self, biometry_config: BiometryConfig):
        self.close()
        # Setup the renderer with the provided biometry configuration
        self._biometry_config = biometry_config
        self._biometric_tracker = BiometricTracker(biometry_config)
        self._audio_alarm = self._prepare_audio_alarm(biometry_config)
        
        # Setup the emotion engine with the provided model path
        self._emotion_engine = self._prepare_emotion_engine(biometry_config)
        
        # Prepare the telemetry HUDs
        self._ocular_telemetry_hud = self._prepare_ocular_telemetry()
        self._oral_telemetry_hud = self._prepare_oral_telemetry()
        self._head_pose_telemetry = self._prepare_head_pose_telemetry()
        self._facial_mesh = self._prepare_facial_mesh()
        self._emotion_telemetry_hud = self._prepare_emotion_telemetry(biometry_config)
        
        # Setup the MediaPipe Face Landmarker
        landmarker_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._biometry_config.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(landmarker_options)
        self.prev_frame_time = time.monotonic()
        self._last_ts_ms = 0

    def close(self) -> None:
        landmarker = getattr(self, "_landmarker", None)
        if landmarker is not None:
            landmarker.close()
            self._landmarker = None
        
    def _is_setup_complete(self) -> bool:
        if self._biometry_config is None or self._biometric_tracker is None or self._landmarker is None:
            return False

        init_objects = [
            self._ocular_telemetry_hud,
            self._oral_telemetry_hud,
            self._head_pose_telemetry,
            self._facial_mesh,
            self._emotion_telemetry_hud,
            self._audio_alarm,
            self._emotion_engine,
        ]
        
        for obj in init_objects:
            if obj is None:
                return False
        
        return True

    def _render_ocular_oral_telemetry(self, frame: FrameCanvas):
        panel_pos = Point(10, 50).scale(frame.ui_scale)
        panel_size = Size(300, 410).scale(frame.ui_scale)

        panel = HUDPanel(panel_pos, panel_size)        
        panel.render(frame, self._current_hud_theme)
        
        pos_offset = Point(10, 10).scale(frame.ui_scale)
        initial_pos = Point(panel_pos.x + pos_offset.x, panel_pos.y + pos_offset.y)
        size_offset = Size(20, 40).scale(frame.ui_scale)
        size = Size(panel_size.width - size_offset.width, (panel_size.height - size_offset.height) // 2) # Size for the time series graph and progress bar
        
        self._ocular_telemetry_hud.label = f"EAR: {self._bio_state.ear.value:.2f}"
        self._ocular_telemetry_hud.render(
            frame,
            initial_pos,
            size,
            self._bio_state.ear.value,
            self._bio_state.ear.history,
            self._current_hud_theme
        )
        
        initial_pos.y += size.height + (10 * (frame.ui_scale * 2))  # Move down for the next graph
        self._oral_telemetry_hud.label = f"MAR: {self._bio_state.mar.value:.2f}"
        self._oral_telemetry_hud.render(
            frame,
            initial_pos,
            size,
            self._bio_state.mar.value,
            self._bio_state.mar.history,
            self._current_hud_theme
        )
        
    def _render_3d_head_pose_telemetry(self, frame: FrameCanvas):
        panel_size = Size(300, 200).scale(frame.ui_scale)
        panel_pos = Point(frame.width - 10 - panel_size.width, 50).scale(frame.ui_scale)

        panel = HUDPanel(panel_pos, panel_size)        
        panel.render(frame, self._current_hud_theme)
        
        pos_offset = Point(10, 30).scale(frame.ui_scale)
        initial_pos = Point(panel_pos.x + pos_offset.x, panel_pos.y + pos_offset.y)
        component_size = Size(panel_size.width - 20, panel_size.height - 40).scale(frame.ui_scale)
        
        cv2.putText(
            frame.canvas,
            "3D HEAD POSE TELEMETRY",
            initial_pos.to_int_tuple(),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * frame.ui_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        self._head_pose_telemetry.render(
            frame,
            self._bio_state,
            initial_pos,
            component_size,
            self._current_hud_theme
        )
        
        self._emotion_telemetry_hud.render(
            frame,
            Point(panel_pos.x, initial_pos.y + component_size.height + (20 * frame.ui_scale)),
            self._emotion_engine.get_confidence_for_emotions(frame, self._bio_state.bbox),
            self._current_hud_theme,
            size=Size(panel_size.width, 150).scale(frame.ui_scale),
        )
        
        
    def _calculate_landmarks(self, frame: cv2.Mat, ts_ms: int):
        result = self._landmarker.detect_for_video(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            ),
            ts_ms,
        )
        return result.face_landmarks[0] if result.face_landmarks else None
    
    def _trigger_audio_alarm(self, now: float):
        state = self._bio_state
        if state.micro_sleep or state.looking_away or state.mar.yawning:
            self._audio_alarm.trigger(now)
        

    def render(self, frame: cv2.Mat):
        if not self._is_setup_complete():
            raise RuntimeError("HUD setup is not complete. Please call setup() before rendering.")
        
        time_now = time.monotonic()
        dt = max(time_now - self.prev_frame_time, 1e-6) if self.prev_frame_time is not None else 0.0
        fps = 1.0 / dt if dt > 0.0 else 0.0
        self.prev_frame_time = time_now
        self._last_ts_ms = max(self._last_ts_ms + 1, int(time_now * 1000))
        
        frame_canvas = FrameCanvas(frame, resize_frame=self.resize_frame)
        
        face_landmarks = self._calculate_landmarks(frame_canvas.canvas, self._last_ts_ms)
        self._bio_state = self._biometric_tracker.update(face_landmarks, frame_canvas.size, time_now)
        self._current_hud_theme = resolve_hud_theme(self._bio_state, time_now)
        
        self._facial_mesh.render(frame_canvas, self._current_hud_theme, self._bio_state, time_now)
        self._render_title_bar(frame_canvas, fps)
        self._render_ocular_oral_telemetry(frame_canvas)
        self._render_3d_head_pose_telemetry(frame_canvas)
        self._trigger_audio_alarm(time_now)
        return frame_canvas.canvas
        