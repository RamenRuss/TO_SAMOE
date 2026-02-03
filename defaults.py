# defaults.py
from __future__ import annotations

from typing import Any, Optional


def pick(value: Any, cfg: Optional[object], attr_path: str, default: Any) -> Any:
    """Утилита для выбора значения из:
    1) явного аргумента (если не None)
    2) cfg по пути attr_path (например "clip_len_frames" или "paths.index_dir")
    3) default

    Это сделано, чтобы функции можно было вызывать и с cfg, и с явными параметрами
    (удобно для экспериментов и тестов).
    """

    if value is not None:
        return value
    if cfg is None:
        return default

    cur: Any = cfg
    for part in attr_path.split("."):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur
