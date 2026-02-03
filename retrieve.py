# retrieve.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from defaults import pick


def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine sim for normalized vectors; if not normalized, it still works but scales vary."""
    return a @ b.T


def frames_to_time(frame_idx: int, fps: Optional[float]) -> Optional[float]:
    if fps is None or fps <= 0:
        return None
    return float(frame_idx) / float(fps)


def retrieve_topk_segments(
    index: Dict[str, Any],
    backend: Any,
    query_text: str,
    *,
    top_k: Optional[int] = None,
    normalize_embeddings: Optional[bool] = None,
    text_emb: Optional[torch.Tensor] = None,
    cfg: Optional[object] = None,
) -> List[Dict[str, Any]]:
    """Возвращает top-k сегментов (по клипам) для query_text.

    Оптимизация для batch:
    - Можно передать заранее посчитанный text_emb (shape [1, D]) и не считать его для каждого видео.
    """

    top_k = int(pick(top_k, cfg, "top_k", 5))
    normalize_embeddings = bool(pick(normalize_embeddings, cfg, "normalize_embeddings", True))

    # 1) text embedding
    if text_emb is None:
        text_emb = backend.encode_text([query_text], normalize=normalize_embeddings)  # [1,D]
    else:
        if text_emb.dim() != 2 or text_emb.shape[0] != 1:
            raise ValueError("text_emb must be [1,D]")

    # 2) similarity
    video_embs: torch.Tensor = index["embeddings"]
    if not isinstance(video_embs, torch.Tensor):
        raise TypeError("index['embeddings'] must be torch.Tensor")

    if video_embs.numel() == 0 or video_embs.shape[0] == 0:
        return []

    # считаем similarity на device, где text_emb (обычно backend.device)
    video_embs = video_embs.to(text_emb.device)
    sims = cosine_sim_matrix(text_emb, video_embs)[0]  # [C]

    k = min(top_k, int(sims.numel()))
    vals, idxs = torch.topk(sims, k=k)

    fps = index.get("fps", None)
    results: List[Dict[str, Any]] = []
    for score, clip_i in zip(vals.tolist(), idxs.tolist()):
        s, e = index["ranges"][clip_i]
        results.append(
            {
                "rank": len(results) + 1,
                "score": float(score),
                "start_frame": int(s),
                "end_frame": int(e),
                "start_time_sec": frames_to_time(s, fps),
                "end_time_sec": frames_to_time(e, fps),
                "segment_path": (index.get("segment_paths")[clip_i] if isinstance(index.get("segment_paths"), list) and clip_i < len(index.get("segment_paths")) else None),
            }
        )
    return results
