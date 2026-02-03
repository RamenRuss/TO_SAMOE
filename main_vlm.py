# main.py
from __future__ import annotations

import glob
from translate import translate_ru_to_en
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
import warnings

import re
# Disable HuggingFace symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Optional: hide other HF/Transformers warnings (можно убрать если не нужно)
warnings.filterwarnings("ignore", message="Recommended: pip install sacremoses.*")
warnings.filterwarnings("ignore", message="Xet Storage is enabled for this repo.*")
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

def has_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text or ""))

import cv2

import batch_run
from cache_io import ensure_all_cache_dirs
from config import parse_cli
from models import create_backend
from run_one_video import process_one_video
from visualize_results import show_top_segments

from split_video_mvp import split_video_mvp
from motion_detect import motion_detect
from person_detect import person_detect
from count_deleteVideo_video import delete_video
from split_video import split_video_web


VIDEO_EXTS = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm")


def fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def _apply_torch_tuning(cfg: object) -> None:
    try:
        import torch
    except Exception:
        return

    mc = getattr(cfg, "model", None)
    num_threads = getattr(mc, "torch_num_threads", None) if mc is not None else None
    if isinstance(num_threads, int) and num_threads > 0:
        try:
            torch.set_num_threads(num_threads)
        except Exception:
            pass


def _segment_path(seg_item) -> str:
    if isinstance(seg_item, (list, tuple)) and len(seg_item) >= 1:
        return str(seg_item[0])
    return str(seg_item)


def _segment_start_time(seg_item, idx: int, segment_seconds: float) -> float:
    if isinstance(seg_item, (list, tuple)) and len(seg_item) >= 2:
        try:
            return float(seg_item[1])
        except Exception:
            pass
    return float(idx) * float(segment_seconds)


def _find_videos_in_dir(root_dir: str) -> List[str]:
    out: List[str] = []
    for ext in VIDEO_EXTS:
        out.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    out = [p for p in out if os.path.isfile(p)]
    out.sort()
    return out


def _pick_roi_once(video_path: str) -> Tuple[int, int, int, int]:
    """Дает пользователю выбрать ROI на ПЕРВОМ кадре исходного видео.

    Управление в окне OpenCV:
      - выделить прямоугольник
      - Enter/Space подтвердить
      - Esc отменить (в этом случае roi будет (0,0,0,0) -> мы бросим ошибку)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео для выбора ROI: {video_path}")

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Не удалось прочитать первый кадр для ROI: {video_path}")

        r = cv2.selectROI(
            "Select ROI (Enter/Space to confirm, Esc to cancel)",
            frame,
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyWindow("Select ROI (Enter/Space to confirm, Esc to cancel)")

        x, y, w, h = [int(v) for v in r]
        if w <= 0 or h <= 0:
            raise ValueError("ROI не выбран (отменено или нулевая область).")
        return x, y, w, h
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def _preprocess_single_video(video_path: str, cfg: object) -> Dict[str, Any]:
    pp = getattr(cfg, "preprocess", None)
    if pp is None or not bool(getattr(pp, "enabled", False)):
        return {"enabled": False, "video_path": video_path}

    if not video_path or not os.path.exists(video_path):
        raise ValueError(f"Видео не найдено: {video_path}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    base_out_path = str(getattr(pp, "splits_root_dir", "video_split"))
    out_dir = os.path.join(base_out_path, video_name)

    if bool(getattr(pp, "clear_out_dir", True)) and os.path.isdir(out_dir):
        import shutil
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    use_roi = bool(getattr(pp, "use_roi", False))
    roi = getattr(pp, "roi", None)

    # ROI выбирается ОДИН РАЗ на исходном видео и применяется ТОЛЬКО в motion/person detect.
    if use_roi and roi is None:
        roi = _pick_roi_once(video_path)

    seg_sec = float(getattr(pp, "segment_seconds", 5.0))

    t0_total = time.perf_counter()

    # ВАЖНО: нарезка ВСЕГДА без crop (ROI не должен влиять на нарезку сегментов)
    segments = split_video_web(
        input_path=video_path,
        out_dir=out_dir,
        segment_seconds=seg_sec,
        roi=roi,
    )

    segments_all = list(segments)
    segments_kept = list(segments)

    motion_cfg = getattr(pp, "motion", None)
    if motion_cfg is not None and bool(getattr(motion_cfg, "enabled", False)):
        for i in range(len(segments_kept) - 1, -1, -1):
            part_path = _segment_path(segments_kept[i])
            ok = motion_detect(
                video_path=part_path,
                min_motion_seconds=float(getattr(motion_cfg, "min_motion_seconds", 3.0)),
                roi=(roi if use_roi else None),
                min_area=int(getattr(motion_cfg, "min_area", 30)),
                threshold_value=int(getattr(motion_cfg, "threshold_value", 13)),
                blur_ksize=int(getattr(motion_cfg, "blur_ksize", 21)),
                max_gap_frames=int(getattr(motion_cfg, "max_gap_frames", 20)),
            )
            if not ok:
                segments_kept.pop(i)

        delete_video(segments_all, segments_kept, folder_path=out_dir)

    person_cfg = getattr(pp, "person", None)
    if person_cfg is not None and bool(getattr(person_cfg, "enabled", False)):
        for i in range(len(segments_kept) - 1, -1, -1):
            part_path = _segment_path(segments_kept[i])
            ok = person_detect(
                video_path=part_path,
                n_frames=int(getattr(person_cfg, "n_frames", 8)),
                roi=(roi if use_roi else None),
                sample_every_n_frames=int(getattr(person_cfg, "sample_every_n_frames", 3)),
                skip_first_n_frames=int(getattr(person_cfg, "skip_first_n_frames", 2)),
                model_path=str(getattr(person_cfg, "model_path", "yolo11s.pt")),
                imgsz=int(getattr(person_cfg, "imgsz", 416)),
                conf=float(getattr(person_cfg, "conf", 0.35)),
                iou=float(getattr(person_cfg, "iou", 0.5)),
                max_seconds=getattr(person_cfg, "max_seconds", None),
                device=getattr(person_cfg, "device", None),
            )
            if not ok:
                segments_kept.pop(i)

        delete_video(segments_all, segments_kept, folder_path=out_dir)

    elapsed = time.perf_counter() - t0_total

    manifest_name = str(getattr(pp, "manifest_name", "segments_manifest.json"))
    manifest_path = os.path.join(out_dir, manifest_name)

    manifest = {
        "source_video": os.path.abspath(video_path),
        "segments_dir": os.path.abspath(out_dir),
        "segment_seconds": seg_sec,
        "use_roi": bool(use_roi),
        "roi": list(roi) if roi is not None else None,
        "segments": [],
    }

    for idx, seg in enumerate(segments_kept):
        seg_path = _segment_path(seg)
        start_sec = _segment_start_time(seg, idx=idx, segment_seconds=seg_sec)
        manifest["segments"].append({"path": os.path.abspath(seg_path), "start_time_sec": float(start_sec)})

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {
        "enabled": True,
        "video_path": video_path,
        "out_dir": out_dir,
        "manifest_path": manifest_path,
        "segments_total": len(segments_all),
        "segments_kept": len(segments_kept),
        "time_sec": elapsed,
        "time_hms": fmt_hms(elapsed),
        "use_roi": bool(use_roi),
        "roi": roi,
    }


def main() -> None:
    run, cfg = parse_cli()

    ensure_all_cache_dirs(cfg=cfg)
    _apply_torch_tuning(cfg)

    pp = getattr(cfg, "preprocess", None)
    preprocess_enabled = bool(getattr(pp, "enabled", False)) if pp is not None else False

    if run.mode == "batch":
        if preprocess_enabled:
            input_root = str(getattr(pp, "input_root_dir", ""))
            if not input_root or not os.path.isdir(input_root):
                raise ValueError(f"preprocess.enabled=True, но preprocess.input_root_dir не папка: {input_root}")

            videos = _find_videos_in_dir(input_root)
            if not videos:
                raise RuntimeError(f"Не найдено видео в папке: {input_root}")

            t0 = time.perf_counter()
            for i, vp in enumerate(videos, 1):
                print(f"\n[Preprocess batch] [{i}/{len(videos)}] {vp}")
                prep = _preprocess_single_video(vp, cfg)
                print(f"  -> kept {prep.get('segments_kept')}/{prep.get('segments_total')} in {prep.get('time_hms')}")

            elapsed = time.perf_counter() - t0
            print(f"\n[Preprocess batch] done in {fmt_hms(elapsed)}")
        splits_root = str(getattr(pp, "splits_root_dir", run.video)) if pp is not None else run.video
        query_batch = run.query
        if has_cyrillic(query_batch):
            query_en = translate_ru_to_en(query_batch)
            if query_en and query_en != query_batch:
                print(f"[Translate] RU -> EN: {query_batch!r} -> {query_en!r}")
            query_batch = query_en
        batch_run.run_batch(videos_dir=splits_root, query=query_batch, cfg=cfg)
        return
    video_path = run.video
    query = run.query
    show = bool(run.show)

    # Auto-translate Russian queries to English for better CLIP/XCLIP retrieval quality
    if has_cyrillic(query):
        query_en = translate_ru_to_en(query)
        if query_en and query_en != query:
            print(f"[Translate] RU -> EN: {query!r} -> {query_en!r}")
        query = query_en

    if not video_path or not os.path.exists(video_path):
        raise ValueError(f"--video не задан или файл не найден: {video_path}")
    if not query:
        raise ValueError("--query не задан")

    presegmented_manifest_path: Optional[str] = None
    if preprocess_enabled:
        prep = _preprocess_single_video(video_path, cfg)
        presegmented_manifest_path = prep.get("manifest_path") if isinstance(prep, dict) else None
        if presegmented_manifest_path:
            print(f"[Preprocess] kept {prep.get('segments_kept')}/{prep.get('segments_total')} segments in {prep.get('time_hms')}")
        else:
            raise RuntimeError("Preprocess включён, но manifest_path не получен")

    backend = create_backend(cfg, load=True)
    cfg_full = asdict(cfg)

    out = process_one_video(
        video_path,
        query=query,
        backend=backend,
        cfg=cfg,
        cfg_full=cfg_full,
        presegmented_manifest_path=presegmented_manifest_path,
    )

    print(f"\n[Single] processing: {out.get('processing_time_hms')} ({out.get('processing_time_sec', 0.0):.2f}s)")

    if show and out.get("results_list"):
        top_k = int(getattr(cfg, "top_k", 5))
        show_top_segments(video_path, out["results_list"], max_clips=top_k, delay=0.15)


if __name__ == "__main__":
    main()