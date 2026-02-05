# index_video.py
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from Utils import pick
from models.base import BaseVideoTextBackend
from Utils import open_video, sample_indices_uniform, build_clips


def index_video_segments(
    video_path: str,
    backend: BaseVideoTextBackend,
    clip_len_frames: Optional[int] = None,
    clip_stride_frames: Optional[int] = None,
    batch_size_clips: Optional[int] = None,
    normalize_embeddings: Optional[bool] = None,
    sample_strategy: Optional[str] = None,
    fps_hint: Optional[float] = None,
    verbose: Optional[bool] = None,
    cfg: Optional[object] = None,
) -> Dict[str, Any]:
    clip_len_frames = int(pick(clip_len_frames, cfg, "clip_len_frames", 8))
    clip_stride_frames = int(pick(clip_stride_frames, cfg, "clip_stride_frames", 64))
    batch_size_clips = int(pick(batch_size_clips, cfg, "batch_size_clips", 32))
    normalize_embeddings = bool(pick(normalize_embeddings, cfg, "normalize_embeddings", True))
    sample_strategy = pick(sample_strategy, cfg, "sample_strategy", "uniform")
    fps_hint = pick(fps_hint, cfg, "fps_hint", None)
    verbose = bool(pick(verbose, cfg, "verbose", True))

    vs = open_video(video_path, fps_hint=fps_hint)
    num_frames = int(vs.num_frames)
    fps = float(vs.fps) if vs.fps is not None else None

    all_ranges = build_clips(num_frames, clip_len_frames, clip_stride_frames)

    decode_sec = 0.0
    encode_sec = 0.0
    batches = 0

    all_embs: List[torch.Tensor] = []

    for b0 in range(0, len(all_ranges), batch_size_clips):
        batch = all_ranges[b0: b0 + batch_size_clips]
        if not batch:
            continue

        batches += 1
        B = len(batch)
        T = clip_len_frames

        frame_idx_flat: List[int] = []
        for (s, e) in batch:
            if sample_strategy == "head":
                idx = list(range(s, min(e, s + T)))
                if len(idx) < T:
                    idx += [idx[-1]] * (T - len(idx)) if idx else [s] * T
            else:
                idx = sample_indices_uniform(s, e, T)
            frame_idx_flat.extend(idx)

        t_dec0 = time.perf_counter()
        frames_flat = vs.get_frames(frame_idx_flat)
        t_dec1 = time.perf_counter()
        decode_sec += (t_dec1 - t_dec0)

        if not isinstance(frames_flat, np.ndarray) or frames_flat.ndim != 4:
            raise RuntimeError("Unexpected frames shape from decoder")

        if frames_flat.shape[0] != B * T:
            raise RuntimeError(f"Unexpected frames shape from decoder: {frames_flat.shape}")

        frames_bt = frames_flat.reshape(B, T, *frames_flat.shape[1:])

        t_enc0 = time.perf_counter()
        embs = backend.encode_video_clips(frames_bt, normalize=normalize_embeddings)
        t_enc1 = time.perf_counter()
        encode_sec += (t_enc1 - t_enc0)

        all_embs.append(embs.detach().float().cpu())

        if verbose and (batches % 10 == 0 or b0 + batch_size_clips >= len(all_ranges)):
            print(f"[Index] {min(b0 + batch_size_clips, len(all_ranges))}/{len(all_ranges)} clips...")

    video_embs = torch.cat(all_embs, dim=0) if all_embs else torch.empty((0, 0), dtype=torch.float32)

    return {
        "video_path": video_path,
        "backend": backend.backend_name,
        "model_name": backend.model_name,
        "fps": fps,
        "num_frames": num_frames,
        "clip_len_frames": clip_len_frames,
        "clip_stride_frames": clip_stride_frames,
        "ranges": all_ranges,
        "embeddings": video_embs,
        "timing": {
            "decode_sec": float(decode_sec),
            "encode_sec": float(encode_sec),
            "batches": int(batches),
        },
    }


def index_presegmented_manifest(
    manifest_path: str,
    backend: BaseVideoTextBackend,
    clip_len_frames: Optional[int] = None,
    batch_size_clips: Optional[int] = None,
    normalize_embeddings: Optional[bool] = None,
    fps_hint: Optional[float] = None,
    verbose: Optional[bool] = None,
    cfg: Optional[object] = None,
) -> Dict[str, Any]:
    import json
    import os

    clip_len_frames = int(pick(clip_len_frames, cfg, "clip_len_frames", 8))
    batch_size_clips = int(pick(batch_size_clips, cfg, "batch_size_clips", 32))
    normalize_embeddings = bool(pick(normalize_embeddings, cfg, "normalize_embeddings", True))
    fps_hint = pick(fps_hint, cfg, "fps_hint", None)
    verbose = bool(pick(verbose, cfg, "verbose", True))

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    segments_dir = manifest.get("segments_dir", None)
    if not segments_dir:
        segments_dir = os.path.dirname(os.path.abspath(manifest_path))
    segments_dir = os.path.abspath(str(segments_dir))

    source_video = manifest.get("source_video", None)
    fps = manifest.get("fps", None)
    try:
        fps = float(fps) if fps is not None else None
    except Exception:
        fps = None
    if fps is None or fps <= 0:
        fps = fps_hint if fps_hint is not None else 30.0

    seg_items = manifest.get("segments", [])
    if not isinstance(seg_items, list):
        seg_items = []

    segment_paths: List[str] = []
    ranges: List[Tuple[int, int]] = []

    all_embs: List[torch.Tensor] = []

    decode_sec = 0.0
    encode_sec = 0.0
    batches = 0

    def _segment_full_path(item: dict) -> str:
        p = item.get("path", None) or item.get("file", None) or item.get("name", None)
        if not p:
            return ""
        p = str(p)
        if os.path.isabs(p):
            return p
        return os.path.join(segments_dir, p)

    def _start_frame(item: dict) -> int:
        t0 = item.get("start_time_sec", 0.0)
        try:
            t0f = float(t0)
        except Exception:
            t0f = 0.0
        return int(round(t0f * float(fps)))

    n = len(seg_items)
    i0 = 0
    while i0 < n:
        batch_items = seg_items[i0: i0 + batch_size_clips]
        i0 += batch_size_clips
        if not batch_items:
            break

        clips_list: List[np.ndarray] = []
        batch_ranges: List[Tuple[int, int]] = []
        batch_paths: List[str] = []

        for it in batch_items:
            if not isinstance(it, dict):
                continue

            seg_path = _segment_full_path(it)
            if not seg_path or not os.path.exists(seg_path):
                continue

            s_frame = _start_frame(it)

            vs = open_video(seg_path, fps_hint=fps_hint)
            num_frames_seg = int(getattr(vs, "num_frames", 0) or 0)
            if num_frames_seg <= 0:
                continue

            t_dec0 = time.perf_counter()
            frame_idx = sample_indices_uniform(0, num_frames_seg, clip_len_frames)
            frames = vs.get_frames(frame_idx)
            t_dec1 = time.perf_counter()
            decode_sec += (t_dec1 - t_dec0)

            if not isinstance(frames, np.ndarray) or frames.ndim != 4 or frames.shape[0] != clip_len_frames:
                continue

            clips_list.append(frames)
            batch_paths.append(seg_path)
            batch_ranges.append((int(s_frame), int(s_frame + num_frames_seg)))

        if not clips_list:
            continue

        clips_bt = np.stack(clips_list, axis=0)

        t_enc0 = time.perf_counter()
        embs = backend.encode_video_clips(clips_bt, normalize=normalize_embeddings)
        t_enc1 = time.perf_counter()
        encode_sec += (t_enc1 - t_enc0)

        all_embs.append(embs.detach().float().cpu())
        segment_paths.extend(batch_paths)
        ranges.extend(batch_ranges)
        batches += 1

        if verbose:
            done = len(segment_paths)
            print(f"[Index/preseg] {done}/{n} segments encoded...")

    video_embs = torch.cat(all_embs, dim=0) if all_embs else torch.empty((0, 0), dtype=torch.float32)

    num_frames_total = 0
    if ranges:
        try:
            num_frames_total = int(max(e for _, e in ranges))
        except Exception:
            num_frames_total = 0

    return {
        "video_path": source_video or "",
        "manifest_path": os.path.abspath(manifest_path),
        "segments_dir": segments_dir,
        "segment_paths": segment_paths,
        "backend": backend.backend_name,
        "model_name": backend.model_name,
        "fps": float(fps),
        "num_frames": int(num_frames_total),
        "clip_len_frames": int(clip_len_frames),
        "clip_stride_frames": 0,
        "ranges": ranges,
        "embeddings": video_embs,
        "timing": {
            "decode_sec": float(decode_sec),
            "encode_sec": float(encode_sec),
            "batches": int(batches),
        },
    }


def index_segment_files(
    segment_paths: List[str],
    backend: BaseVideoTextBackend,
    *,
    clip_len_frames: Optional[int] = None,
    batch_size_clips: Optional[int] = None,
    normalize_embeddings: Optional[bool] = None,
    fps_hint: Optional[float] = None,
    verbose: Optional[bool] = None,
    cfg: Optional[object] = None,
) -> Dict[str, Any]:
    clip_len_frames = int(pick(clip_len_frames, cfg, "clip_len_frames", 8))
    batch_size_clips = int(pick(batch_size_clips, cfg, "batch_size_clips", 32))
    normalize_embeddings = bool(pick(normalize_embeddings, cfg, "normalize_embeddings", True))
    fps_hint = pick(fps_hint, cfg, "fps_hint", None)
    verbose = bool(pick(verbose, cfg, "verbose", True))

    segs = [p for p in (segment_paths or []) if isinstance(p, str) and p]
    segs = [p for p in segs]

    decode_sec = 0.0
    encode_sec = 0.0
    batches = 0

    all_embs: List[torch.Tensor] = []
    ok_paths: List[str] = []
    ranges: List[Tuple[int, int]] = []

    n = len(segs)
    i0 = 0
    while i0 < n:
        batch = segs[i0: i0 + batch_size_clips]
        i0 += batch_size_clips
        if not batch:
            break

        clips_list: List[np.ndarray] = []
        batch_paths: List[str] = []
        batch_ranges: List[Tuple[int, int]] = []

        for seg_path in batch:
            vs = open_video(seg_path, fps_hint=fps_hint)
            num_frames_seg = int(getattr(vs, "num_frames", 0) or 0)
            if num_frames_seg <= 0:
                continue

            t_dec0 = time.perf_counter()
            frame_idx = sample_indices_uniform(0, num_frames_seg, clip_len_frames)
            frames = vs.get_frames(frame_idx)
            t_dec1 = time.perf_counter()
            decode_sec += (t_dec1 - t_dec0)

            if not isinstance(frames, np.ndarray) or frames.ndim != 4 or frames.shape[0] != clip_len_frames:
                continue

            clips_list.append(frames)
            batch_paths.append(seg_path)
            batch_ranges.append((0, int(num_frames_seg)))

        if not clips_list:
            continue

        clips_bt = np.stack(clips_list, axis=0)

        t_enc0 = time.perf_counter()
        embs = backend.encode_video_clips(clips_bt, normalize=normalize_embeddings)
        t_enc1 = time.perf_counter()
        encode_sec += (t_enc1 - t_enc0)

        all_embs.append(embs.detach().float().cpu())
        ok_paths.extend(batch_paths)
        ranges.extend(batch_ranges)
        batches += 1

        if verbose:
            done = len(ok_paths)
            print(f"[Index/segments] {done}/{n} segments encoded...")

    video_embs = torch.cat(all_embs, dim=0) if all_embs else torch.empty((0, 0), dtype=torch.float32)

    return {
        "video_path": "",
        "backend": backend.backend_name,
        "model_name": backend.model_name,
        "fps": None,
        "num_frames": 0,
        "clip_len_frames": int(clip_len_frames),
        "clip_stride_frames": 0,
        "ranges": ranges,
        "segment_paths": ok_paths,
        "embeddings": video_embs,
        "timing": {
            "decode_sec": float(decode_sec),
            "encode_sec": float(encode_sec),
            "batches": int(batches),
        },
    }
