from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from ironmanhudv2.emotion_engine.cnn_fer import CNNFer
from ironmanhudv2.emotion_engine.ensemble_fer import EnsembleFER
from ironmanhudv2.frame_canvas import FrameCanvas
from ironmanhudv2.math import BBox


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
        use_tta: bool | None = None,
    ):
        model_path = self._resolve_model_path(Path(torch_model))
        self.torch_model = model_path
        self.img_size = img_size
        self.device = self._get_torch_device(device)

        map_location = self.device if self.device.type == "cuda" else "cpu"
        checkpoint = torch.load(model_path, map_location=map_location, weights_only=True)
        self.class_names: list[str] = self._load_class_names(checkpoint, model_path)
        self.use_tta: bool = (
            bool(use_tta)
            if use_tta is not None
            else checkpoint.get("tta") == "mean_of_original_and_horizontal_flip_probabilities"
        )
        self.model = self._load_model(checkpoint)

    @staticmethod
    def _load_class_names(checkpoint: dict, model_path: Path) -> list[str]:
        class_names = checkpoint.get("class_names")
        if class_names is None:
            class_names_path = model_path.parent / "class_names.json"
            if class_names_path.is_file():
                with class_names_path.open("r", encoding="utf-8") as f:
                    class_names = json.load(f)

        if not class_names:
            raise ValueError(f"Missing class_names in checkpoint {model_path} and no class_names.json found.")

        return [str(name) for name in class_names]

    def _build_single_model(self, arch: dict | None, state_dict: dict) -> nn.Module:
        arch_cfg = dict(arch or {})
        arch_cfg.setdefault("n_classes", len(self.class_names))

        if "backbone_name" in arch_cfg:
            from ironmanhudv2.emotion_engine.transfer_fer import TransferFER

            # Force pretrained=False so inference never attempts to download weights.
            arch_cfg["pretrained"] = False
            model: nn.Module = TransferFER(**arch_cfg)
        else:
            model = CNNFer(**arch_cfg)

        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()
        return model

    def _load_single_model(self, checkpoint: dict) -> nn.Module:
        if "model_state_dict" not in checkpoint:
            raise ValueError("Single-model checkpoint is missing 'model_state_dict'.")
        return self._build_single_model(checkpoint.get("arch"), checkpoint["model_state_dict"])

    def _load_ensemble_model(self, checkpoint: dict) -> EnsembleFER:
        ensemble_method = checkpoint.get("ensemble_method", "mean_of_model_probabilities")
        if ensemble_method != "mean_of_model_probabilities":
            raise ValueError(f"Unsupported ensemble_method: {ensemble_method}")

        models_dict = checkpoint.get("models")
        if not isinstance(models_dict, dict) or not models_dict:
            raise ValueError("Ensemble checkpoint must define a non-empty 'models' mapping.")

        submodels: list[nn.Module] = []
        for sub_name, sub_ckpt in models_dict.items():
            if not isinstance(sub_ckpt, dict) or "model_state_dict" not in sub_ckpt:
                raise ValueError(f"Ensemble submodel '{sub_name}' is missing 'model_state_dict'.")
            submodels.append(self._build_single_model(sub_ckpt.get("arch"), sub_ckpt["model_state_dict"]))

        model = EnsembleFER(submodels).to(self.device)
        model.eval()
        return model

    def _load_model(self, checkpoint: dict) -> nn.Module:
        if "models" in checkpoint:
            return self._load_ensemble_model(checkpoint)
        return self._load_single_model(checkpoint)

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
            if isinstance(self.model, EnsembleFER):
                probs_tensor = self.model.forward_probs(tensor, use_tta=self.use_tta)
            else:
                probs_tensor = torch.softmax(self.model(tensor), dim=1)
                if self.use_tta:
                    probs_flip = torch.softmax(self.model(torch.flip(tensor, dims=[-1])), dim=1)
                    probs_tensor = 0.5 * (probs_tensor + probs_flip)
            probabilities = probs_tensor[0].detach().cpu().numpy()

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
        