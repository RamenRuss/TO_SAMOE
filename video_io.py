# video_io.py
from __future__ import annotations

"""Видео I/O через decord.

Ключевое ускорение относительно предыдущей версии:
- Не создаём VideoReader на каждый клип.
- Держим VideoReader один раз и читаем пачки кадров через get_batch().

Это сильно сокращает overhead декодирования/инициализации и ускоряет индексацию.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import numpy as np


def _try_import_decord():
    try:
        import decord
        from decord import VideoReader
        return decord, VideoReader
    except Exception as e:
        raise ImportError(
            "Не найден decord. Установи: pip install decord\n"
            f"Текущая ошибка: {e}"
        )


@dataclass
class VideoSource:
    """Обёртка над decord.VideoReader с удобным batch API."""

    video_path: str
    fps_hint: Optional[float] = None

    def __post_init__(self) -> None:
        decord, VideoReader = _try_import_decord()
        # context по умолчанию cpu; можно расширить под gpu позже
        self._decord = decord
        self._vr = VideoReader(self.video_path)

        self.num_frames: int = len(self._vr)

        fps: Optional[float] = None
        try:
            fps = float(self._vr.get_avg_fps())
        except Exception:
            fps = self.fps_hint

        self.fps: Optional[float] = fps

    def get_frames(self, frame_indices: List[int]) -> np.ndarray:
        """Возвращает RGB uint8 кадры [T,H,W,3] по индексам."""

        if self.num_frames <= 0:
            return np.zeros((0, 0, 0, 3), dtype=np.uint8)

        max_idx = self.num_frames - 1
        safe_idx = [min(max(int(i), 0), max_idx) for i in frame_indices]
        frames = self._vr.get_batch(safe_idx).asnumpy()  # [T,H,W,3] uint8 RGB
        return frames


# ----------------------------
# Backward-compatible helpers
# ----------------------------

def open_video(video_path: str, fps_hint: Optional[float] = None) -> VideoSource:
    return VideoSource(video_path=video_path, fps_hint=fps_hint)


def read_video_metadata(video_path: str, fps_hint: Optional[float] = None) -> dict:
    vs = open_video(video_path, fps_hint=fps_hint)
    return {"num_frames": vs.num_frames, "fps": vs.fps}


def read_frames(video_path: str, frame_indices: List[int]) -> np.ndarray:
    """Старый API (медленный, т.к. открывает VideoReader каждый вызов).

    Оставлен для совместимости (визуализация).
    В индексации использовать VideoSource.
    """

    vs = open_video(video_path)
    return vs.get_frames(frame_indices)


# ----------------------------
# Sampling / clips
# ----------------------------

def sample_indices_uniform(start: int, end: int, num: int) -> List[int]:
    """Uniform sample `num` indices in [start, end) (end exclusive)."""

    if end <= start:
        return [start] * num
    if num == 1:
        return [start]
    lin = np.linspace(start, end - 1, num=num)
    return [int(round(x)) for x in lin]


def build_clips(frame_count: int, clip_len: int, stride: int) -> List[Tuple[int, int]]:
    """Возвращает список клипов как (start_frame, end_frame_exclusive)."""

    clips: List[Tuple[int, int]] = []
    start = 0
    while start < frame_count:
        end = start + clip_len
        if end > frame_count:
            end = frame_count
        clips.append((start, end))
        if start + stride >= frame_count:
            break
        start += stride
    return clips
