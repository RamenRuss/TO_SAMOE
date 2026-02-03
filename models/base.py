# models/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class BackendInfo:
    backend: str
    model_name: str
    embedding_dim: int
    notes: str = ""


class BaseVideoTextBackend:
    """Базовый интерфейс для video-text retrieval бэкендов.

    Требования к бэкенду:
    - load(): загрузка модели/процессора
    - encode_text(): эмбеддинг текста [N, D]
    - encode_video_clips(): эмбеддинг клипов [B, D]

    Почему так:
    - Индексация и retrieval должны быть независимы от конкретной модели.
    - Чтобы добавить новую модель — создаёшь файл models/<name>_backend.py и регистрируешь класс.
    """

    backend_name: str = "base"

    def __init__(self, cfg: object):
        self.cfg = cfg
        # Поля должны быть заполнены в load()
        self.model_name: str = ""
        self.device: str = "cpu"
        self.dtype_str: str = "fp32"
        self._loaded: bool = False

    def load(self) -> None:  # pragma: no cover
        raise NotImplementedError

    @property
    def loaded(self) -> bool:
        return bool(self._loaded)

    def encode_text(self, texts: List[str], *, normalize: bool = True) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def encode_video_clips(self, clips_rgb_uint8: np.ndarray, *, normalize: bool = True) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    # ----------------------------
    # Cache fingerprints
    # ----------------------------

    def fingerprint(self) -> Dict[str, Any]:
        """То, что реально влияет на эмбеддинги.

        По умолчанию: backend + model_name.
        Если у бэкенда есть дополнительные параметры (например pooling),
        он должен переопределить fingerprint().
        """

        return {
            "backend": self.backend_name,
            "model_name": self.model_name,
        }

    def info(self) -> BackendInfo:
        """Инфо для логов/отчётов."""
        return BackendInfo(
            backend=self.backend_name,
            model_name=self.model_name,
            embedding_dim=-1,
            notes="",
        )
