# batch_run.py
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from cache_io import ensure_all_cache_dirs
from config import parse_cli
from index_video import index_segment_files
from models import create_backend


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def _iter_media_files(root: Path, *, recursive: bool) -> List[str]:
    if not root.exists():
        return []
    if root.is_file() and root.suffix.lower() in VIDEO_EXTS:
        return [str(root)]
    if not root.is_dir():
        return []

    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    else:
        files = [p for p in root.glob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return [str(p) for p in sorted(files)]


def _load_manifests_segments(
    folder: str,
    *,
    recursive: bool,
    manifest_name: str,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    root = Path(folder)
    if not root.exists():
        return [], {}

    if root.is_file():
        return [], {}

    if recursive:
        manifests = [p for p in root.rglob("*") if p.is_file() and p.name == manifest_name]
    else:
        manifests = [p for p in root.glob("*") if p.is_file() and p.name == manifest_name]

    seg_paths: List[str] = []
    meta_map: Dict[str, Dict[str, Any]] = {}

    for mp in sorted(manifests):
        try:
            with open(mp, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            continue

        if not isinstance(manifest, dict):
            continue

        segments_dir = manifest.get("segments_dir") or str(mp.parent)
        try:
            segments_dir = str(Path(segments_dir).resolve())
        except Exception:
            segments_dir = str(mp.parent.resolve())

        source_video = manifest.get("source_video")
        segment_seconds = manifest.get("segment_seconds", None)
        use_roi = manifest.get("use_roi", None)
        roi = manifest.get("roi", None)

        items = manifest.get("segments", [])
        if not isinstance(items, list):
            continue

        for it in items:
            if not isinstance(it, dict):
                continue

            fn = it.get("path") or it.get("file") or it.get("name")
            if not fn:
                continue

            p = Path(fn)
            if not p.is_absolute():
                p = Path(segments_dir) / p
            p = p.resolve()

            if not p.exists() or p.suffix.lower() not in VIDEO_EXTS:
                continue

            start_time_sec = it.get("start_time_sec", None)
            end_time_sec = it.get("end_time_sec", None)
            try:
                if end_time_sec is None and start_time_sec is not None and segment_seconds is not None:
                    end_time_sec = float(start_time_sec) + float(segment_seconds)
            except Exception:
                pass

            seg_path = str(p)
            seg_paths.append(seg_path)

            meta_map[seg_path] = {
                "segment_path": seg_path,
                "segment_file": p.name,
                "segments_dir": segments_dir,
                "manifest_path": str(mp.resolve()),
                "source_video": source_video,
                "source_video_file": (Path(str(source_video)).name if source_video else None),
                "start_time_sec": start_time_sec,
                "end_time_sec": end_time_sec,
                "segment_seconds": segment_seconds,
                "use_roi": use_roi,
                "roi": roi,
            }

    seg_paths = sorted(set(seg_paths))
    return seg_paths, meta_map


def _merge_segment_meta(
    seg_path: str,
    score: float,
    rank: int,
    meta_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    base = {
        "rank": int(rank),
        "score": float(score),
        "segment_path": seg_path,
    }
    m = meta_map.get(seg_path)
    if not m:
        return base

    out = dict(base)
    for k in (
        "source_video",
        "source_video_file",
        "start_time_sec",
        "end_time_sec",
        "segment_seconds",
        "manifest_path",
        "segments_dir",
        "segment_file",
        "use_roi",
        "roi",
    ):
        if k in m:
            out[k] = m.get(k)

    return out


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _save_segment_file(src_path: str, dst_path: str, mode: str) -> None:
    _ensure_parent_dir(dst_path)
    mode = (mode or "copy").lower().strip()

    if os.path.exists(dst_path):
        return

    if mode == "hardlink":
        os.link(src_path, dst_path)
        return

    if mode == "symlink":
        os.symlink(src_path, dst_path)
        return

    shutil.copy2(src_path, dst_path)


def run_batch(
    *,
    videos_dir: str,
    query: str,
    cfg: object,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_all_cache_dirs(cfg=cfg)

    backend = create_backend(cfg, load=True)

    out_dir = out_dir or getattr(cfg, "batch_out_dir", "./batch_out")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    summary_path = str(Path(out_dir) / "summary.json")
    report_path = str(Path(out_dir) / "report.json")

    batch_cfg = getattr(cfg, "batch", None)
    recursive = True
    if batch_cfg is not None:
        recursive = bool(getattr(batch_cfg, "recursive", True))

    save_top_segments = False
    save_top_segments_dir = "selected_segments"
    save_top_segments_mode = "copy"
    if batch_cfg is not None:
        save_top_segments = bool(getattr(batch_cfg, "save_top_segments", False))
        save_top_segments_dir = str(getattr(batch_cfg, "save_top_segments_dir", "selected_segments"))
        save_top_segments_mode = str(getattr(batch_cfg, "save_top_segments_mode", "copy"))

    manifest_name = None
    pp = getattr(cfg, "preprocess", None)
    if pp is not None:
        manifest_name = getattr(pp, "manifest_name", None)
    manifest_name = str(manifest_name) if manifest_name else "segments_manifest.json"

    seg_paths, meta_map = _load_manifests_segments(videos_dir, recursive=recursive, manifest_name=manifest_name)

    if not seg_paths:
        root = Path(videos_dir)
        seg_paths = _iter_media_files(root, recursive=recursive)
        meta_map = {}
        for p in seg_paths:
            meta_map[p] = {
                "segment_path": p,
                "segment_file": Path(p).name,
                "segments_dir": str(Path(p).parent),
                "manifest_path": None,
                "source_video": None,
                "source_video_file": None,
                "start_time_sec": None,
                "end_time_sec": None,
                "segment_seconds": None,
                "use_roi": None,
                "roi": None,
            }

    if not seg_paths:
        raise RuntimeError(f"No segment videos found in: {videos_dir}")

    t0 = time.perf_counter()

    normalize_embeddings = bool(getattr(cfg, "normalize_embeddings", True))
    text_emb = backend.encode_text([query], normalize=normalize_embeddings)

    index = index_segment_files(seg_paths, backend=backend, cfg=cfg)

    embs: torch.Tensor = index["embeddings"]
    indexed_paths: List[str] = index.get("segment_paths", []) or []

    summary: List[Dict[str, Any]] = []

    if isinstance(embs, torch.Tensor) and embs.numel() > 0 and indexed_paths:
        embs = embs.to(text_emb.device)
        sims = (text_emb @ embs.T)[0]

        top_k = int(getattr(cfg, "top_k", 5))
        k = min(top_k, int(sims.numel()))
        vals, idxs = torch.topk(sims, k=k)

        for rank, (score, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), 1):
            seg_path = indexed_paths[idx] if idx < len(indexed_paths) else None
            if not seg_path:
                continue
            summary.append(_merge_segment_meta(seg_path, score=float(score), rank=rank, meta_map=meta_map))

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    selected_dir = ""
    saved_selected: List[str] = []
    save_error: Optional[str] = None

    if save_top_segments and summary:
        try:
            selected_dir = str(Path(out_dir) / save_top_segments_dir)
            Path(selected_dir).mkdir(parents=True, exist_ok=True)

            for item in summary:
                seg_path = item.get("segment_path")
                if not seg_path or not os.path.exists(seg_path):
                    continue

                rank = item.get("rank", 0)
                score = item.get("score", None)

                src_name = Path(seg_path).name
                score_s = "na"
                try:
                    score_s = f"{float(score):.6f}"
                except Exception:
                    pass

                dst_name = f"rank{int(rank):04d}_score{score_s}_{src_name}"
                dst_path = str(Path(selected_dir) / dst_name)

                _save_segment_file(str(seg_path), dst_path, mode=save_top_segments_mode)
                saved_selected.append(dst_path)

        except Exception as e:
            save_error = repr(e)
            selected_dir = ""
            saved_selected = []

    elapsed = time.perf_counter() - t0

    report = {
        "videos_dir": videos_dir,
        "query": query,
        "backend": backend.backend_name,
        "model_name": backend.model_name,
        "recursive": bool(recursive),
        "manifest_name": manifest_name,
        "num_segments_found": len(seg_paths),
        "num_segments_indexed": len(indexed_paths),
        "top_k": int(getattr(cfg, "top_k", 5)),
        "batch_wall_time_sec": float(elapsed),
        "batch_wall_time_hms": fmt_hms(elapsed),
        "out_dir": out_dir,
        "summary_path": summary_path,
        "save_top_segments": bool(save_top_segments),
        "save_top_segments_dir": selected_dir,
        "save_top_segments_mode": save_top_segments_mode,
        "saved_top_segments": saved_selected,
    }
    if save_error is not None:
        report["save_top_segments_error"] = save_error

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Summary: {summary_path}")
    print(f"Report:  {report_path}")
    print(f"Segments indexed: {report['num_segments_indexed']}/{report['num_segments_found']}")
    if save_top_segments:
        print(f"Saved top segments: {len(saved_selected)} -> {selected_dir}" if selected_dir else "Saved top segments: failed")
    print(f"Batch wall time: {report['batch_wall_time_hms']}")

    return report


def main() -> None:
    run, cfg = parse_cli()
    if run.mode != "batch":
        raise SystemExit("batch_run.py запускается только с --mode batch")

    pp = getattr(cfg, "preprocess", None)
    default_dir = getattr(pp, "splits_root_dir", None) if pp is not None else None
    videos_dir = run.video if run.video else (str(default_dir) if default_dir else "")

    if not videos_dir:
        raise ValueError("Не задан путь к папке фрагментов: укажи --video или cfg.preprocess.splits_root_dir")

    run_batch(videos_dir=videos_dir, query=run.query, cfg=cfg)


if __name__ == "__main__":
    main()
