# models/dummy_backend.py
from __future__ import annotations

"""Dummy backend.

Нужен для:
- быстрых smoke-тестов структуры проекта без скачивания реальных моделей
- проверки, что индексация/кэш/retrieval работают end-to-end на синтетических данных

ВАЖНО: это НЕ настоящая модель, качество = 0, только для тестов.
"""

import hashlib
from typing import Any, Dict, List

import numpy as np
import torch

from .base import BaseVideoTextBackend, BackendInfo
from .registry import register_backend


def _hash_to_vec(seed: bytes, dim: int = 512) -> torch.Tensor:
    h = hashlib.sha256(seed).digest()
    # повторяем байты до нужной длины
    raw = (h * ((dim * 4 // len(h)) + 1))[: dim * 4]
    arr = np.frombuffer(raw, dtype=np.uint32).astype(np.float32)
    arr = arr[:dim]
    v = torch.from_numpy(arr)
    v = v / (v.norm(p=2) + 1e-9)
    return v


@register_backend
class DummyBackend(BaseVideoTextBackend):
    backend_name = "dummy"

    def load(self) -> None:
        self.model_name = "dummy"
        self.device = "cpu"
        self.dtype_str = "fp32"
        self._loaded = True

    def encode_text(self, texts: List[str], *, normalize: bool = True) -> torch.Tensor:
        vecs = []
        for t in texts:
            v = _hash_to_vec(("text:" + t).encode("utf-8"), dim=512)
            vecs.append(v)
        out = torch.stack(vecs, dim=0)
        return out

    def encode_video_clips(self, clips_rgb_uint8: np.ndarray, *, normalize: bool = True) -> torch.Tensor:
        # clips_rgb_uint8: [B,T,H,W,3]
        if clips_rgb_uint8.ndim != 5:
            raise ValueError("clips_rgb_uint8 must be [B,T,H,W,3]")
        B = clips_rgb_uint8.shape[0]
        vecs = []
        for i in range(B):
            # берём средний цвет как "сигнал", а затем хэшируем
            m = float(np.mean(clips_rgb_uint8[i]))
            v = _hash_to_vec((f"video:{m:.6f}" + str(clips_rgb_uint8[i].shape)).encode("utf-8"), dim=512)
            vecs.append(v)
        return torch.stack(vecs, dim=0)

    def info(self) -> BackendInfo:
        return BackendInfo(backend=self.backend_name, model_name=self.model_name, embedding_dim=512, notes="dummy")

    def fingerprint(self) -> Dict[str, Any]:
        return {"backend": self.backend_name, "model_name": self.model_name, "dim": 512}
