# models/clip_frame_backend.py
from __future__ import annotations

"""CLIP-frame backend (быстрый).

Идея:
- Для каждого клипа берём T кадров (как и раньше).
- Считаем image features для каждого кадра CLIP'ом.
- Агрегируем кадры в эмбеддинг клипа (mean pooling).
- Сравниваем с text features (CLIP text encoder).

Плюсы:
- Очень быстрый по сравнению с video-трансформерами (особенно на CPU).
- Простой и стабильный.

Минусы:
- Это НЕ "настоящая" video-language модель: хуже на событиях/действиях,
  где важна динамика, а не статичные кадры.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base import BaseVideoTextBackend, BackendInfo
from .registry import register_backend


def _ensure_tensor(x: Any, *, name: str = "output") -> torch.Tensor:
    """Привести выход HF-модели к torch.Tensor.

    Некоторые модели/обёртки возвращают transformers.ModelOutput
    (например BaseModelOutputWithPooling). Эта функция извлекает
    нужный тензор из типичных полей.
    """
    if isinstance(x, torch.Tensor):
        return x
    # common HF fields
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(x, attr):
            v = getattr(x, attr)
            if isinstance(v, torch.Tensor):
                # last_hidden_state: берём CLS токен
                if attr == "last_hidden_state" and v.dim() >= 2:
                    return v[:, 0]
                return v
    # tuple/list: первый элемент часто tensor
    if isinstance(x, (tuple, list)) and x and isinstance(x[0], torch.Tensor):
        return x[0]
    raise TypeError(f"{name} must be torch.Tensor, got {type(x)}")



def _torch_dtype(dtype_str: str) -> torch.dtype:
    return torch.float16 if str(dtype_str).lower() == "fp16" else torch.float32


@register_backend
class ClipFrameBackend(BaseVideoTextBackend):
    backend_name = "clip_frame"

    def __init__(self, cfg: object):
        super().__init__(cfg)
        self.model = None
        self.processor = None

    def load(self) -> None:
        # imports here -> чтобы dummy backend работал без transformers
        try:
            from transformers import CLIPModel, CLIPProcessor
        except Exception as e:
            raise ImportError(
                "Не удалось импортировать transformers.CLIPModel/CLIPProcessor. "
                "Установи transformers. Ошибка: " + repr(e)
            )

        mc = getattr(self.cfg, "model", None)
        backend_default = "openai/clip-vit-base-patch32"

        model_name = getattr(mc, "model_name", None) or backend_default
        device = getattr(mc, "device", "cpu")
        dtype_str = getattr(mc, "dtype", "fp32")
        use_compile = bool(getattr(mc, "use_torch_compile", False))

        self.model_name = str(model_name)
        self.device = str(device)
        self.dtype_str = str(dtype_str)

        dtype = _torch_dtype(self.dtype_str)

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)

        if hasattr(self.model, "config"):
            self.model.config.return_dict = True

        self.model.to(self.device)
        # fp16 имеет смысл только на cuda
        if str(self.device).startswith("cuda") and dtype == torch.float16:
            self.model = self.model.to(dtype=dtype)

        self.model.eval()

        if use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self._loaded = True

    @torch.no_grad()
    def encode_text(self, texts: List[str], *, normalize: bool = True) -> torch.Tensor:
        if not self.loaded:
            raise RuntimeError("Backend not loaded")

        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        feats = self.model.get_text_features(**inputs)  # [N,D]
        feats = _ensure_tensor(feats, name="get_text_features")
        if normalize:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats

    @torch.no_grad()
    def encode_video_clips(self, clips_rgb_uint8: np.ndarray, *, normalize: bool = True) -> torch.Tensor:
        if not self.loaded:
            raise RuntimeError("Backend not loaded")

        if clips_rgb_uint8.ndim != 5 or clips_rgb_uint8.shape[-1] != 3:
            raise ValueError(f"clips_rgb_uint8 must be [B,T,H,W,3], got {clips_rgb_uint8.shape}")

        if clips_rgb_uint8.dtype != np.uint8:
            clips_rgb_uint8 = clips_rgb_uint8.astype(np.uint8, copy=False)

        B, T, H, W, C = clips_rgb_uint8.shape

        # CLIPProcessor иногда умеет принимать batched numpy сразу, пробуем это (быстрее).
        frames_bt = clips_rgb_uint8.reshape(B * T, H, W, C)
        try:
            out = self.processor(images=frames_bt, return_tensors="pt")
        except Exception:
            frames: List[np.ndarray] = [frames_bt[i] for i in range(frames_bt.shape[0])]
            out = self.processor(images=frames, return_tensors="pt")
        pixel_values: torch.Tensor = out["pixel_values"]  # [B*T,3,H',W']

        pixel_values = pixel_values.to(self.device)

        # Чанкуем по кадрам, чтобы не упереться в память
        frame_bs = int(getattr(getattr(self.cfg, "model", None), "frame_batch_size", 256) or 256)
        frame_bs = max(1, frame_bs)

        feats_list: List[torch.Tensor] = []
        for chunk in torch.split(pixel_values, frame_bs, dim=0):
            feats = self.model.get_image_features(pixel_values=chunk)  # [n,D]
            feats = _ensure_tensor(feats, name="get_image_features")
            feats_list.append(feats)

        frame_feats = torch.cat(feats_list, dim=0)  # [B*T, D]
        frame_feats = frame_feats.view(B, T, -1)    # [B, T, D]

        # mean pooling по кадрам
        clip_feats = frame_feats.mean(dim=1)        # [B, D]

        if normalize:
            clip_feats = torch.nn.functional.normalize(clip_feats, dim=-1)

        return clip_feats

    def info(self) -> BackendInfo:
        # embedding_dim: можно достать из config
        dim = -1
        try:
            dim = int(getattr(self.model.config, "projection_dim", -1))
        except Exception:
            dim = -1
        return BackendInfo(
            backend=self.backend_name,
            model_name=self.model_name,
            embedding_dim=dim,
            notes="CLIP per-frame mean pooling",
        )

    def fingerprint(self) -> Dict[str, Any]:
        # на эмбеддинги влияет model + pooling стратегия
        return {
            "backend": self.backend_name,
            "model_name": self.model_name,
            "pool": "mean",
        }