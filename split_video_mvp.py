import os
import math
import cv2


def split_video_mvp(
    input_path: str,
    out_dir: str,
    segment_seconds: float,
    crop: bool,
    *,
    roi: tuple[int, int, int, int] | None = None,
) -> list[tuple[str, float]]:
    """
    Разбивает видео на части фиксированной длительности.

    ВАЖНО (по архитектуре проекта):
    - ROI НЕ используется на этапе нарезки.
    - ROI задаётся 1 раз на исходном видео и применяется ТОЛЬКО в motion_detect / person_detect.
    - Поэтому split_video_mvp всегда режет полные кадры (crop=False).

    Возвращает:
        list[(out_path, start_time_seconds)]
    """

    if crop:
        # Жёстко запрещаем — иначе ROI "лезет" не туда и ломает пайплайн.
        raise ValueError(
            "split_video_mvp: crop=True запрещён. ROI применяется только в motion/person detect. "
            "Установи crop=False."
        )

    # Создаём выходную папку
    os.makedirs(out_dir, exist_ok=True)

    # Открываем видео
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {input_path}")

    # FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    fps = float(fps)

    if segment_seconds <= 0:
        cap.release()
        raise ValueError("segment_seconds должен быть > 0")

    frames_per_seg = int(round(fps * segment_seconds))
    frames_per_seg = max(1, frames_per_seg)

    video_name = os.path.splitext(os.path.basename(input_path))[0]

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    try:
        total_frames_i = int(total_frames) if total_frames is not None else 0
    except Exception:
        total_frames_i = 0

    if total_frames_i > 0:
        est_segments = int(math.ceil(total_frames_i / float(frames_per_seg)))
        digits = max(6, len(str(max(est_segments - 1, 0))))
    else:
        digits = 6

    # Читаем первый кадр, чтобы узнать размеры
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError(f"Не удалось прочитать первый кадр: {input_path}")

    src_h, src_w = first_frame.shape[:2]
    out_w, out_h = int(src_w), int(src_h)

    # Возвращаемся на начало (первый кадр уже прочитан)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    segments: list[tuple[str, float]] = []
    seg_idx = 0

    writer: cv2.VideoWriter | None = None

    def open_new_segment(start_frame_idx: int):
        nonlocal writer, seg_idx

        if writer is not None:
            writer.release()

        out_path = os.path.join(out_dir, f"{video_name}_{seg_idx:0{digits}d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer_ = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
        if not writer_.isOpened():
            raise RuntimeError(f"Не удалось создать VideoWriter: {out_path}")

        writer = writer_

        start_time_sec = float(start_frame_idx) / float(fps)
        segments.append((out_path, start_time_sec))
        seg_idx += 1

    frame_idx = 0
    open_new_segment(start_frame_idx=0)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_idx > 0 and (frame_idx % frames_per_seg == 0):
                open_new_segment(start_frame_idx=frame_idx)

            # Пишем полный кадр (никакого ROI/crop)
            if writer is not None:
                writer.write(frame)

            frame_idx += 1

    finally:
        if writer is not None:
            writer.release()
        cap.release()

    return segments
