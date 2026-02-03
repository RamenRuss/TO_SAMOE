# models/xclip_backend.py
from __future__ import annotations

"""XCLIP backend (video-text).

Это ближе к "правильной" video-language модели, чем frame-based CLIP.
Но заметно тяжелее, особенно на CPU.

Для ускорения (при сохранении video-text модели) разумный старт:
- backend = xclip
- model_name = microsoft/xclip-base-patch32 (обычно быстрее patch16)
- device = cuda:0 + dtype = fp16 (если есть GPU)

Источники:
- модель microsoft/xclip-base-patch32 на HF
- модель microsoft/xclip-base-patch16 на HF
"""

from typing import Any, Dict, List

import numpy as np
import torch

from .base import BaseVideoTextBackend, BackendInfo
from .registry import register_backend


def _ensure_tensor(x: Any, *, name: str = "output") -> torch.Tensor:
    """Привести выход HF-модели к torch.Tensor (см. clip_frame_backend)."""
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(x, attr):
            v = getattr(x, attr)
            if isinstance(v, torch.Tensor):
                if attr == "last_hidden_state" and v.dim() >= 2:
                    return v[:, 0]
                return v
    if isinstance(x, (tuple, list)) and x and isinstance(x[0], torch.Tensor):
        return x[0]
    raise TypeError(f"{name} must be torch.Tensor, got {type(x)}")



def _torch_dtype(dtype_str: str) -> torch.dtype:
    return torch.float16 if str(dtype_str).lower() == "fp16" else torch.float32


def _squeeze_singletons(pixel_values: torch.Tensor) -> torch.Tensor:
    """Убираем лишние оси размера 1, если processor их добавил."""
    while pixel_values.dim() > 4 and 1 in pixel_values.shape:
        for d in range(pixel_values.dim()):
            if pixel_values.shape[d] == 1:
                pixel_values = pixel_values.squeeze(d)
                break
    return pixel_values


@register_backend
class XClipBackend(BaseVideoTextBackend):
    backend_name = "xclip"

    def __init__(self, cfg: object):
        super().__init__(cfg)
        self.model = None
        self.processor = None

    def load(self) -> None:
        try:
            from transformers import XCLIPModel, XCLIPProcessor
        except Exception as e:
            raise ImportError(
                "Не удалось импортировать transformers.XCLIPModel/XCLIPProcessor. "
                "Установи transformers. Ошибка: " + repr(e)
            )

        mc = getattr(self.cfg, "model", None)
        backend_default = "microsoft/xclip-base-patch32"  # быстрее, чем patch16

        model_name = getattr(mc, "model_name", None) or backend_default
        device = getattr(mc, "device", "cpu")
        dtype_str = getattr(mc, "dtype", "fp32")
        use_compile = bool(getattr(mc, "use_torch_compile", False))

        self.model_name = str(model_name)
        self.device = str(device)
        self.dtype_str = str(dtype_str)

        dtype = _torch_dtype(self.dtype_str)

        self.processor = XCLIPProcessor.from_pretrained(self.model_name)
        self.model = XCLIPModel.from_pretrained(self.model_name)

        if hasattr(self.model, "config"):
            self.model.config.return_dict = True

        self.model.to(self.device)
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

        feats = self.model.get_text_features(**inputs)
        feats = _ensure_tensor(feats, name="get_text_features")
        feats = _ensure_tensor(feats, name="video_feats")

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

        frames_bt = clips_rgb_uint8.reshape(B * T, H, W, C)
        try:
            out = self.processor(images=frames_bt, return_tensors="pt")
        except Exception:
            frames: List[np.ndarray] = [frames_bt[i] for i in range(frames_bt.shape[0])]
            out = self.processor(images=frames, return_tensors="pt")
        pixel_values: torch.Tensor = out["pixel_values"]  # [B*T,3,H',W'] обычно

        pixel_values = _squeeze_singletons(pixel_values)
        if pixel_values.dim() != 4:
            raise RuntimeError(f"pixel_values must be 4D [N,3,H,W], got {tuple(pixel_values.shape)}")

        # Пробуем быстрый/корректный путь через get_video_features (если доступно)
        pv = pixel_values.view(B, T, *pixel_values.shape[1:])  # [B,T,3,H',W']
        pv = pv.to(self.device)

        feats = None
        if hasattr(self.model, "get_video_features"):
            try:
                feats = self.model.get_video_features(pixel_values=pv)  # [B,D]
                feats = _ensure_tensor(feats, name="get_video_features")
            except Exception:
                feats = None

        if feats is None:
            # Фоллбек: vision_model -> visual_projection -> mit
            pv_flat = pv.view(B * T, *pv.shape[2:])  # [B*T,3,H,W]

            vision_out = self.model.vision_model(pixel_values=pv_flat, return_dict=False)
            if isinstance(vision_out, (tuple, list)):
                last_hidden = vision_out[0]
                pooled = vision_out[1] if len(vision_out) > 1 and vision_out[1] is not None else last_hidden[:, 0]
            else:
                last_hidden = vision_out.last_hidden_state
                pooled = vision_out.pooler_output if vision_out.pooler_output is not None else last_hidden[:, 0]

            if not hasattr(self.model, "visual_projection") or self.model.visual_projection is None:
                raise RuntimeError("model.visual_projection missing")

            frame_feats = self.model.visual_projection(pooled)  # [B*T, 512] обычно
            frame_feats = frame_feats.view(B, T, -1)

            mit_out = self.model.mit(frame_feats)
            if isinstance(mit_out, (tuple, list)):
                mit_last = mit_out[0]
                mit_pooled = mit_out[1] if len(mit_out) > 1 and mit_out[1] is not None else mit_last[:, 0]
            else:
                mit_last = mit_out.last_hidden_state
                mit_pooled = mit_out.pooler_output if getattr(mit_out, "pooler_output", None) is not None else mit_last[:, 0]

            feats = mit_pooled  # [B, 512]

        feats = _ensure_tensor(feats, name="video_feats")

        if normalize:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats

    def info(self) -> BackendInfo:
        dim = -1
        try:
            dim = int(getattr(self.model.config, "projection_dim", -1))
        except Exception:
            dim = -1
        return BackendInfo(
            backend=self.backend_name,
            model_name=self.model_name,
            embedding_dim=dim,
            notes="XCLIP video-text",
        )

    def fingerprint(self) -> Dict[str, Any]:
        # На эмбеддинги влияет модель + способ агрегации (здесь: встроенный MIT)
        return {
            "backend": self.backend_name,
            "model_name": self.model_name,
            "video": True,
        }