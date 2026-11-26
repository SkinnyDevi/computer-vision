from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ironmanhudv2.emotion_engine.cnn_fer import CNNFer
from ironmanhudv2.frame_canvas import FrameCanvas
from ironmanhudv2.hud_panel import HUDPanel
from ironmanhudv2.math import BBox
from ironmanhudv2.units import Point, Size


@dataclass
class EmotionPrediction:
    emotion: str
    confidence: float
    scores: dict[str, float]


class EmotionEngine:
    def _get_torch_device(self, device: str | torch.device | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(
        self,
        torch_model: Path | str,
        device: str | torch.device | None = None,
        img_size: int = 48,
    ):
        model_path = self._resolve_model_path(Path(torch_model))
        self.torch_model = model_path
        self.img_size = img_size
        self.device = self._get_torch_device(device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        class_names = checkpoint.get("class_names")
        if class_names is None:
            class_names_path = model_path.parent / "class_names.json"
            if class_names_path.is_file():
                with class_names_path.open("r", encoding="utf-8") as f:
                    class_names = json.load(f)
        
        if not class_names:
            raise ValueError(f"Missing class_names in checkpoint {model_path} and no class_names.json found.")

        self.class_names: list[str] = [str(name) for name in class_names]

        arch = dict(checkpoint.get("arch", {}))
        arch.setdefault("n_classes", len(self.class_names))

        self.model = CNNFer(**arch).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()

    @staticmethod
    def _resolve_model_path(path: Path) -> Path:
        if path.is_dir():
            pt_files = sorted(path.glob("*.pt"))
            if not pt_files:
                raise FileNotFoundError(f"No .pt model checkpoint found in directory: {path}")
            return pt_files[0]
        
        if path.is_file():
            return path
        
        raise FileNotFoundError(f"Torch model path does not exist: {path}")

    @staticmethod
    def _crop_face_roi(img: np.ndarray, bbox: BBox, pad: float = 0.12) -> np.ndarray | None:
        if bbox.width <= 0 or bbox.height <= 0:
            return None
        
        h, w = img.shape[:2]
        px = int(bbox.width * pad)
        py = int(bbox.height * pad)
        x1 = max(0, bbox.x1 - px)
        y1 = max(0, bbox.y1 - py)
        x2 = min(w, bbox.x2 + px)
        y2 = min(h, bbox.y2 + py)
        if x2 <= x1 or y2 <= y1:
            return None
        
        roi = img[y1:y2, x1:x2]
        return roi if roi.size > 0 else None

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        if img.ndim == 2:
            gray = img
        elif img.ndim == 3 and img.shape[2] == 1:
            gray = img[:, :, 0]
        elif img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 3 and img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported image shape for emotion inference: {img.shape}")

        h, w = gray.shape[:2]
        if (h, w) != (self.img_size, self.img_size):
            interp = cv2.INTER_AREA if (h >= self.img_size and w >= self.img_size) else cv2.INTER_LINEAR
            gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=interp)

        arr = np.ascontiguousarray(gray, dtype=np.uint8)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)
    
    def _predict(
        self,
        frame: FrameCanvas,
        bbox: BBox | None = None,
        pad: float = 0.12,
    ):
        img = frame.canvas
        if img is None or img.size == 0:
            return None

        roi = self._crop_face_roi(img, bbox, pad=pad) if bbox is not None else img
        if roi is None or roi.size == 0:
            return None

        tensor = self._preprocess(roi)
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        scores = {label: float(prob) for label, prob in zip(self.class_names, probabilities)}
        return scores, probabilities
        
    def make_emotion_prediction(
        self,
        frame: FrameCanvas,
        bbox: BBox | None = None,
        pad: float = 0.12,
    ) -> EmotionPrediction | None:
        result = self._predict(frame, bbox=bbox, pad=pad)
        if result is None:
            return None
        
        scores, probabilities = result
        best_idx = int(np.argmax(probabilities))
        return EmotionPrediction(
            emotion=self.class_names[best_idx],
            confidence=float(probabilities[best_idx]),
            scores=scores,
        )
        
    def get_confidence_for_emotions(
        self,
        frame: FrameCanvas,
        bbox: BBox | None = None,
        emotions: list[str] = None,
        pad: float = 0.12,
    ) -> list[EmotionPrediction]:
        if not emotions:
            emotions = self.class_names
        
        result = self._predict(frame, bbox=bbox, pad=pad)
        if result is None:
            return []
        
        scores, probabilities = result
        predictted_emotions: list[EmotionPrediction] = []
        for emotion in emotions:    
            idx = self.class_names.index(emotion)
            confidence = float(probabilities[idx])
            predictted_emotions.append(EmotionPrediction(emotion=emotion, confidence=confidence, scores=scores))
            
        return predictted_emotions
        