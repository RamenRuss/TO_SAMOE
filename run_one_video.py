# run_one_video.py
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from cache_io import (
    ensure_all_cache_dirs,
    make_index_path,
    make_results_path,
    save_index,
    load_index,
    index_matches_cfg,
    save_results_json,
    load_results_json,
    stable_hash,
    results_matches_cfg,
)
from index_video import index_video_segments, index_presegmented_manifest
from Utils import retrieve_topk_segments
from models.base import BaseVideoTextBackend


def safe_name(s: str, max_len: int = 90) -> str:
    s = re.sub(r"[^\w\-\.\(\) ]+", "_", s, flags=re.UNICODE).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len] if len(s) > max_len else s


def fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def extract_top_times(results_list: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    n = max(0, int(n))
    out: List[Dict[str, Any]] = []
    for i, r in enumerate((results_list or [])[:n], 1):
        out.append(
            {
                "rank": r.get("rank", i),
                "score": r.get("score"),
                "start_time_sec": r.get("start_time_sec"),
                "end_time_sec": r.get("end_time_sec"),
                "start_frame": r.get("start_frame"),
                "end_frame": r.get("end_frame"),
                "segment_path": r.get("segment_path"),
            }
        )
    return out


def _index_fingerprint(cfg: object, backend: BaseVideoTextBackend) -> Dict[str, Any]:
    return {
        "backend": backend.fingerprint(),
        "clip_len_frames": int(getattr(cfg, "clip_len_frames", 0)),
        "clip_stride_frames": int(getattr(cfg, "clip_stride_frames", 0)),
        "sample_strategy": getattr(cfg, "sample_strategy", None),
        "normalize_embeddings": bool(getattr(cfg, "normalize_embeddings", True)),
        "fps_hint": getattr(cfg, "fps_hint", None),
        "batch_size_clips": int(getattr(cfg, "batch_size_clips", 0)),
    }


def _results_fingerprint(cfg: object, backend: BaseVideoTextBackend) -> Dict[str, Any]:
    return {
        "index": _index_fingerprint(cfg, backend),
        "backend": backend.fingerprint(),
        "normalize_embeddings": bool(getattr(cfg, "normalize_embeddings", True)),
    }


_FFMPEG_OK: Optional[bool] = None


def ensure_ffmpeg_tools() -> None:
    global _FFMPEG_OK
    if _FFMPEG_OK is True:
        return
    if _FFMPEG_OK is False:
        raise FileNotFoundError("ffmpeg/ffprobe not available")

    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["ffprobe", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _FFMPEG_OK = True
    except Exception as e:
        _FFMPEG_OK = False
        raise FileNotFoundError(f"ffmpeg/ffprobe not available: {e}")


def video_duration_sec_ffprobe(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    s = (p.stdout or b"").decode("utf-8", errors="ignore").strip()
    try:
        return float(s)
    except Exception:
        return 0.0


def export_segment_ffmpeg(video_path: str, out_path: str, start_sec: float, end_sec: float, *, reencode: bool = True) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    start_sec = float(start_sec)
    end_sec = float(end_sec)
    dur = max(0.0, end_sec - start_sec)
    if dur <= 0:
        raise ValueError(f"Non-positive duration: {start_sec}..{end_sec}")

    if reencode:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            video_path,
            "-t",
            f"{dur:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            out_path,
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            video_path,
            "-t",
            f"{dur:.3f}",
            "-c",
            "copy",
            out_path,
        ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def export_topk_clips(
    video_path: str,
    query: str,
    results_list: List[Dict[str, Any]],
    out_dir: str,
    *,
    top_k: int = 5,
    pad_sec: float = 2.0,
    reencode: bool = True,
    skip_existing: bool = True,
    force: bool = False,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    q = safe_name(query)

    saved: List[str] = []
    for i, r in enumerate((results_list or [])[:top_k], 1):
        t0 = r.get("start_time_sec", None)
        t1 = r.get("end_time_sec", None)
        if t0 is None or t1 is None:
            continue

        start = max(0.0, float(t0) - pad_sec)
        end = max(start, float(t1) + pad_sec)

        score = r.get("score", None)
        score_s = "na" if score is None else f"{float(score):.4f}"

        out_name = f"rank{i}_score{score_s}_{q}.mp4"
        out_path = os.path.join(out_dir, out_name)

        if skip_existing and (not force) and os.path.exists(out_path):
            saved.append(out_path)
            continue

        export_segment_ffmpeg(video_path, out_path, start, end, reencode=reencode)
        saved.append(out_path)

    return saved


def process_one_video(
    video_path: str,
    *,
    query: str,
    backend: BaseVideoTextBackend,
    cfg: object,
    presegmented_manifest_path: Optional[str] = None,
    cfg_full: Optional[dict] = None,
    clips_root: Optional[str] = None,
    text_emb: Optional[Any] = None,
) -> Dict[str, Any]:
    ensure_all_cache_dirs(cfg=cfg)

    if cfg_full is None:
        try:
            cfg_full = asdict(cfg)
        except Exception:
            cfg_full = {"cfg": str(cfg)}

    t_meta0 = time.perf_counter()
    dur_sec = 0.0
    try:
        ensure_ffmpeg_tools()
        dur_sec = video_duration_sec_ffprobe(video_path)
    except Exception:
        dur_sec = 0.0
    t_meta1 = time.perf_counter()

    stage: Dict[str, Any] = {}

    def _mark(name: str, t0: float, t1: float) -> None:
        stage[f"{name}_sec"] = float(t1 - t0)
        stage[f"{name}_hms"] = fmt_hms(stage[f"{name}_sec"])

    _mark("video_meta", t_meta0, t_meta1)

    top_k = int(getattr(cfg, "top_k", 5))
    save_top_n_times = int(getattr(cfg, "save_top_n_times", top_k))

    cache_index = bool(getattr(cfg, "cache_index", True))
    strict_cache_match = bool(getattr(cfg, "strict_cache_match", True))
    force_reindex = bool(getattr(cfg, "force_reindex", False))

    save_results = bool(getattr(cfg, "save_results", True)) if hasattr(cfg, "save_results") else bool(getattr(cfg, "cache_results", True))
    use_cached_results = bool(getattr(cfg, "use_results_cache", False))
    export_clips = bool(getattr(cfg, "export_clips", False))

    export_top_k = int(getattr(cfg, "export_top_k", top_k))
    pad_sec = float(getattr(cfg, "pad_sec", 2.0))
    reencode = bool(getattr(cfg, "reencode", True))

    index_fp = _index_fingerprint(cfg, backend)
    results_fp = _results_fingerprint(cfg, backend)

    manifest_hash = None
    manifest_payload = None
    if presegmented_manifest_path:
        import json

        with open(presegmented_manifest_path, "r", encoding="utf-8") as f:
            manifest_payload = json.load(f)
        manifest_hash = stable_hash(manifest_payload)
        index_fp = dict(index_fp)
        index_fp["presegmented"] = True
        index_fp["manifest_hash"] = manifest_hash
        results_fp = dict(results_fp)
        results_fp["index"] = index_fp
        results_fp["presegmented"] = True
        results_fp["manifest_hash"] = manifest_hash

    index_path = make_index_path(
        video_path=video_path,
        backend_name=backend.backend_name,
        model_name=backend.model_name,
        index_fingerprint=index_fp,
        cfg=cfg,
    )
    results_path = make_results_path(
        video_path=video_path,
        backend_name=backend.backend_name,
        model_name=backend.model_name,
        query=query,
        results_fingerprint=results_fp,
        cfg=cfg,
    )

    t_total0 = time.perf_counter()

    t_cache0 = time.perf_counter()
    if use_cached_results and os.path.exists(results_path):
        payload = load_results_json(results_path)
        ok = (not strict_cache_match) or results_matches_cfg(payload, results_fp)
        if ok:
            full_list = payload.get("results_list", []) if isinstance(payload, dict) else []
            if isinstance(full_list, list) and len(full_list) >= top_k:
                results_list = full_list[:top_k]
                top_times = payload.get("top_times") if isinstance(payload, dict) else None
                if not top_times:
                    top_times = extract_top_times(results_list, save_top_n_times)

                t_cache1 = time.perf_counter()
                _mark("cache_check", t_cache0, t_cache1)

                elapsed = time.perf_counter() - t_total0
                stage["total_sec"] = float(elapsed)
                stage["total_hms"] = fmt_hms(elapsed)

                return {
                    "video": video_path,
                    "presegmented_manifest_path": presegmented_manifest_path,
                    "manifest_hash": manifest_hash,
                    "query": query,
                    "backend": backend.backend_name,
                    "model_name": backend.model_name,
                    "cached_results": True,
                    "cached_index": None,
                    "index_path": index_path,
                    "results_path": results_path,
                    "results_list": results_list,
                    "top_times": top_times,
                    "saved_clips": [],
                    "clips_dir": "",
                    "video_duration_sec": dur_sec,
                    "video_duration_hms": fmt_hms(dur_sec),
                    "processing_time_sec": float(elapsed),
                    "processing_time_hms": fmt_hms(elapsed),
                    "stage_times": stage,
                }

    t_cache1 = time.perf_counter()
    _mark("cache_check", t_cache0, t_cache1)

    t_index0 = time.perf_counter()

    index: Optional[Dict[str, Any]] = None
    cached_index = False

    if cache_index and os.path.exists(index_path) and not force_reindex:
        idx_payload = load_index(index_path)
        if (not strict_cache_match) or index_matches_cfg(idx_payload, index_fp):
            index = idx_payload
            cached_index = True

    index_profile = {"decode_sec_total": 0.0, "encode_sec_total": 0.0, "clips": 0, "batches": 0}

    if index is None:
        if presegmented_manifest_path:
            index = index_presegmented_manifest(presegmented_manifest_path, backend=backend, cfg=cfg)
        else:
            index = index_video_segments(video_path, backend=backend, cfg=cfg)
        timing = index.get("timing", {}) if isinstance(index, dict) else {}
        index_profile["decode_sec_total"] = float(timing.get("decode_sec", 0.0))
        index_profile["encode_sec_total"] = float(timing.get("encode_sec", 0.0))
        index_profile["batches"] = int(timing.get("batches", 0))
        index_profile["clips"] = len(index.get("ranges", [])) if isinstance(index, dict) else 0

        if cache_index:
            save_index(index_path, index=index, index_fingerprint=index_fp)

    t_index1 = time.perf_counter()
    _mark("index_build_or_load", t_index0, t_index1)

    t_ret0 = time.perf_counter()
    results_list = retrieve_topk_segments(index, backend=backend, query_text=query, cfg=cfg, text_emb=text_emb)
    results_list = (results_list or [])[:top_k]
    t_ret1 = time.perf_counter()
    _mark("retrieve", t_ret0, t_ret1)

    top_times = extract_top_times(results_list, save_top_n_times)

    t_save0 = time.perf_counter()
    if save_results:
        save_results_json(
            results_path,
            {
                "video": video_path,
                "presegmented_manifest_path": presegmented_manifest_path,
                "manifest_hash": manifest_hash,
                "query": query,
                "backend": backend.backend_name,
                "model_name": backend.model_name,
                "cfg_full": cfg_full,
                "index_fingerprint": index_fp,
                "results_fingerprint": results_fp,
                "cfg_hash": stable_hash(results_fp),
                "results_list": results_list,
                "top_times": top_times,
            },
        )
    t_save1 = time.perf_counter()
    _mark("save_results", t_save0, t_save1)

    t_exp0 = time.perf_counter()
    saved_clips: List[str] = []
    clips_dir = ""
    export_error: Optional[str] = None

    if export_clips:
        try:
            ensure_ffmpeg_tools()

            if clips_root is None:
                out_dir = getattr(cfg, "batch_out_dir", "./batch_out")
                clips_subdir = getattr(cfg, "clips_subdir", "clips")
                clips_root = os.path.join(out_dir, clips_subdir)

            video_base = safe_name(os.path.splitext(os.path.basename(video_path))[0])
            clips_dir = os.path.join(clips_root, video_base, safe_name(query))
            saved_clips = export_topk_clips(
                video_path=video_path,
                query=query,
                results_list=results_list,
                out_dir=clips_dir,
                top_k=min(export_top_k, top_k),
                pad_sec=pad_sec,
                reencode=reencode,
                skip_existing=True,
                force=False,
            )
        except Exception as e:
            export_error = repr(e)
            saved_clips = []
            clips_dir = ""

    t_exp1 = time.perf_counter()
    _mark("export", t_exp0, t_exp1)

    elapsed = time.perf_counter() - t_total0
    stage["total_sec"] = float(elapsed)
    stage["total_hms"] = fmt_hms(elapsed)

    out: Dict[str, Any] = {
        "video": video_path,
        "presegmented_manifest_path": presegmented_manifest_path,
        "manifest_hash": manifest_hash,
        "query": query,
        "backend": backend.backend_name,
        "model_name": backend.model_name,
        "cached_results": False,
        "cached_index": cached_index,
        "index_path": index_path,
        "results_path": results_path,
        "results_list": results_list,
        "top_times": top_times,
        "saved_clips": saved_clips,
        "clips_dir": clips_dir,
        "top1": results_list[0] if results_list else None,
        "video_duration_sec": dur_sec,
        "video_duration_hms": fmt_hms(dur_sec),
        "processing_time_sec": float(elapsed),
        "processing_time_hms": fmt_hms(elapsed),
        "stage_times": stage,
        "index_profile": index_profile,
    }
    if export_error is not None:
        out["export_error"] = export_error

    return out
