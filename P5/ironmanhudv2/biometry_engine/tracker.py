import time

import cv2
import numpy as np
from collections import deque 

from ironmanhudv2.biometry_engine.biometric_state import BiometricState
from ironmanhudv2.biometry_engine.ear_mar import compute_EAR, compute_MAR
from ironmanhudv2.biometry_engine.engine_config import BiometryConfig
from ironmanhudv2.biometry_engine.mediapipe_constants import LEFT_EYE_IDXS, MODEL_3D_POINTS, PNP_LANDMARK_IDXS, RIGHT_EYE_IDXS
from ironmanhudv2.math import compute_bbox, exp_moving_avg
from ironmanhudv2.units import Size


class BiometricTracker:
    def __init__(self, config: BiometryConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._ear_ema: float | None = None
        self._mar_ema: float | None = None
        self._pose_ema: np.ndarray | None = None
        self._pose_offset = np.zeros(3, dtype=np.float64)
        self._last_raw_pose = np.zeros(3, dtype=np.float64)
        self._has_valid_pose: bool = False
        self._is_calibrated: bool = False

        self._closed_frames = 0
        self._eye_closed_start: float | None = None
        self._blink_timestamps: deque = deque()
        self.blink_count = 0

        self._mouth_open_start: float | None = None
        self._yawn_counted = False
        self.yawn_count = 0

        self._nod_start: float | None = None
        self._away_start: float | None = None
        self.ear_history: deque = deque(maxlen=self.cfg.history_len)
        self.mar_history: deque = deque(maxlen=self.cfg.history_len)

    def extract_landmarks_2d(self, face_landmarks, size: Size) -> np.ndarray:
        return np.array([[lm.x * size.width, lm.y * size.height] for lm in face_landmarks], dtype=np.float64)

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate_pose(self) -> np.ndarray | None:
        if not self._has_valid_pose:
            return None

        self._pose_offset = self._last_raw_pose.copy()
        self._pose_ema = np.zeros(3, dtype=np.float64)
        self._nod_start = None
        self._away_start = None
        self._is_calibrated = True
        return self._pose_offset.copy()
    
    def _update_blink_counter(self, state: BiometricState, now: float) -> None:
        if state.ear.raw < self.cfg.params_EAR.threshold:
            self._closed_frames += 1
            if self._eye_closed_start is None:
                self._eye_closed_start = now
                
            state.ear.eye_closed = True
            state.ear.eye_closed_duration = now - self._eye_closed_start
            state.micro_sleep = state.ear.eye_closed_duration >= self.cfg.params_EAR.micro_sleep_seconds
        else:
            if self._closed_frames >= self.cfg.params_EAR.consec_frames_blink:
                self.blink_count += 1
                self._blink_timestamps.append(now)
                
            self._closed_frames = 0
            self._eye_closed_start = None

        while self._blink_timestamps and (now - self._blink_timestamps[0]) > 60.0:
            self._blink_timestamps.popleft()
            
        state.ear.blink_count = self.blink_count
        state.ear.blink_rate_bpm = float(len(self._blink_timestamps))

    def _update_ocular_metrics(self, state: BiometricState, pts_2d: np.ndarray, now: float) -> None:
        state.ear.left_eye = compute_EAR(pts_2d, LEFT_EYE_IDXS)
        state.ear.right_eye = compute_EAR(pts_2d, RIGHT_EYE_IDXS)
        state.ear.raw = 0.5 * (state.ear.left_eye + state.ear.right_eye)
        
        self._ear_ema = exp_moving_avg(self._ear_ema, state.ear.raw, self.cfg.params_EAR.exp_moving_avg_alpha)
        state.ear.value = float(self._ear_ema)
        
        self.ear_history.append(state.ear.value)
        self._update_blink_counter(state, now)

    def _update_oral_metrics(self, state: BiometricState, pts_2d: np.ndarray, now: float) -> None:
        state.mar.raw = compute_MAR(pts_2d)
        self._mar_ema = exp_moving_avg(self._mar_ema, state.mar.raw, self.cfg.params_MAR.exp_moving_avg_alpha)
        state.mar.value = float(self._mar_ema)
        self.mar_history.append(state.mar.value)

        if state.mar.value > self.cfg.params_MAR.threshold:
            if self._mouth_open_start is None:
                self._mouth_open_start = now
                
            state.mar.mouth_open = True
            state.mar.mouth_open_duration = now - self._mouth_open_start
            
            if state.mar.mouth_open_duration >= self.cfg.params_MAR.yawn_seconds:
                state.mar.yawning = True
                if not self._yawn_counted:
                    self.yawn_count += 1
                    self._yawn_counted = True
        else:
            self._mouth_open_start = None
            self._yawn_counted = False
            
        state.mar.yawn_count = self.yawn_count
        
    def build_camera_matrix(self, size: Size) -> np.ndarray:
        focal = float(size.width)
        return np.array(
            [
                [focal, 0.0, size.width / 2.0],
                [0.0, focal, size.height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def rotation_matrix_to_euler(self, rmat: np.ndarray) -> tuple[float, float, float]:
        '''Convierte matriz de rotación a (pitch, yaw, roll) en grados desde el eje frontal nasal.'''
        
        # TODO: Consider using cv2.decomposeProjectionMatrix for a more robust solution
        # TODO: Find simpler way to convert rotation matrix to Euler angles without using Rodrigues
        nose_fwd = rmat @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        pitch = float(np.degrees(np.arcsin(np.clip(-nose_fwd[1], -1.0, 1.0))))
        yaw = float(np.degrees(np.arctan2(nose_fwd[0], -nose_fwd[2])))
        roll = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
        return pitch, yaw, roll
    
    def solve_head_pose_pnp(
        self, pts_2d: np.ndarray, size: Size
    ) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''Estima rotación/traslación 3D de la cabeza mediante cv2.solvePnP.'''
        
        image_points = pts_2d[PNP_LANDMARK_IDXS].astype(np.float64)
        cam_matrix = self.build_camera_matrix(size)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            MODEL_3D_POINTS, 
            image_points, 
            cam_matrix, 
            dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok or tvec[2, 0] < 0:
            _, rvec, tvec = cv2.solvePnP(
                MODEL_3D_POINTS, 
                image_points, 
                cam_matrix, 
                dist_coeffs, 
                flags=cv2.SOLVEPNP_EPNP
            )

        rmat, _ = cv2.Rodrigues(rvec) # Rodrigues: Converts rotation vector to rotation matrix
        pitch, yaw, roll = self.rotation_matrix_to_euler(rmat)
        return pitch, yaw, roll, rvec, tvec, cam_matrix, dist_coeffs

    def _update_attention_flags(self, state: BiometricState, now: float) -> None:
        if state.pitch.value <= self.cfg.params_pose3D.pitch_nod_threshold:
            if self._nod_start is None:
                self._nod_start = now
                
            state.nod_duration = now - self._nod_start
            state.nodding_off = state.nod_duration >= self.cfg.params_pose3D.pose_alert_seconds
        else:
            self._nod_start = None

        is_away = abs(state.yaw.value) >= self.cfg.params_pose3D.yaw_look_away_threshold or state.pitch.value >= self.cfg.params_pose3D.pitch_up_threshold
        if is_away:
            if self._away_start is None:
                self._away_start = now
            
            state.looking_away_duration = now - self._away_start
            state.looking_away = state.looking_away_duration >= self.cfg.params_pose3D.pose_alert_seconds
        else:
            self._away_start = None

    def _update_pose_metrics(self, state: BiometricState, pts_2d: np.ndarray, size: Size, now: float) -> None:
        raw_p, raw_y, raw_r, rvec, tvec, cam_mat, dist = self.solve_head_pose_pnp(pts_2d, size)
        raw_vec = np.array([raw_p, raw_y, raw_r], dtype=np.float64)
        self._last_raw_pose = raw_vec
        self._has_valid_pose = True
        
        rel_pose = ((raw_vec - self._pose_offset + 180.0) % 360.0) - 180.0
        self._pose_ema = exp_moving_avg(
            self._pose_ema, 
            rel_pose, 
            self.cfg.params_pose3D.exp_moving_avg_alpha
        )

        state.pitch.raw, state.yaw.raw, state.roll.raw = raw_p, raw_y, raw_r
        state.pitch.value, state.yaw.value, state.roll.value = map(float, self._pose_ema)
        state.rvec, state.tvec, state.cam_matrix, state.dist_coeffs = rvec, tvec, cam_mat, dist
        self._update_attention_flags(state, now)

    def update(self, face_landmarks, frame_size: Size, now: float | None = None) -> BiometricState:
        now = time.time() if now is None else now
        state = BiometricState(timestamp=now)
        state.ear.blink_count = self.blink_count
        state.mar.yawn_count = self.yawn_count

        if not face_landmarks:
            self._has_valid_pose = False
            self.ear_history.append(0.0)
            self.mar_history.append(0.0)
        else:
            pts_2d = self.extract_landmarks_2d(face_landmarks, frame_size)
            state.face_detected = True
            state.pts_2d = pts_2d
            state.bbox = compute_bbox(pts_2d, frame_size)
            
            self._update_ocular_metrics(state, pts_2d, now)
            self._update_oral_metrics(state, pts_2d, now)
            self._update_pose_metrics(state, pts_2d, frame_size, now)

        state.ear.history = list(self.ear_history)
        state.mar.history = list(self.mar_history)
        return state