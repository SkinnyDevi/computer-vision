from dataclasses import dataclass, field

@dataclass
class ParamEAR:
    threshold: float = 0.21
    consec_frames_blink: int = 2
    micro_sleep_seconds: float = 1.2
    exp_moving_avg_alpha: float = 0.55
    
@dataclass
class ParamMAR:
    threshold: float = 0.55
    yawn_seconds: float = 1.0
    exp_moving_avg_alpha: float = 0.45
    
@dataclass
class ParamPose3D:
    pitch_nod_threshold: float = -18.0
    pitch_up_threshold: float = 25.0
    yaw_look_away_threshold: float = 28.0
    pose_alert_seconds: float = 1.0
    exp_moving_avg_alpha: float = 0.40

@dataclass
class BiometryConfig:
    model_path: str = "face_landmarker.task"
    emotion_model_path: str = "cnn_fer_model.pt"
    custom_emotion_model_path: str | None = None
    use_deepface: bool = False  # Activar True para fusión asíncrona con DeepFace

    params_EAR: ParamEAR = field(default_factory=ParamEAR)
    params_MAR: ParamMAR = field(default_factory=ParamMAR)
    params_pose3D: ParamPose3D = field(default_factory=ParamPose3D)

    emotion_conf_threshold: float = 0.5

    # Alarma sonora e historial visual
    alarm_wav_path: str = "jarvis_alarm.wav"
    alarm_cooldown_seconds: float = 1.8
    history_len: int = 72