# models/registry.py
from __future__ import annotations

from typing import Dict, List, Type

from .base import BaseVideoTextBackend


_BACKENDS: Dict[str, Type[BaseVideoTextBackend]] = {}


def register_backend(cls: Type[BaseVideoTextBackend]) -> Type[BaseVideoTextBackend]:
    name = getattr(cls, "backend_name", None)
    if not name:
        raise ValueError("Backend class must define backend_name")
    if name in _BACKENDS:
        raise ValueError(f"Backend '{name}' is already registered")
    _BACKENDS[name] = cls
    return cls


def available_backends() -> List[str]:
    return sorted(_BACKENDS.keys())


def create_backend(cfg: object, *, load: bool = True) -> BaseVideoTextBackend:
    backend_name = getattr(getattr(cfg, "model", None), "backend", None) or ""
    backend_name = str(backend_name).strip().lower()

    cls = _BACKENDS.get(backend_name)
    if cls is None:
        raise ValueError(
            f"Unknown backend '{backend_name}'. Available: {', '.join(available_backends())}"
        )

    backend = cls(cfg)
    if load:
        backend.load()
    return backend
