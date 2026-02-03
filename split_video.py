import os
import cv2


def split_video_web(
    input_path: str,
    out_dir: str,
    segment_seconds: float = 5,
    *,
    roi: tuple[int, int, int, int] | None = None,  # (x, y, w, h) в пикселях исходного видео
    force_resize_to_first_frame: bool = True,
) -> list[tuple[str, float]]:
    """
    Разбивает видео на части фиксированной длительности.
    При roi != None — обрезает кадры по прямоугольнику ROI (web-выбор).
    Никаких окон выбора области.

    Возвращает:
        list[(out_path, start_time_seconds)]
    """

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frames_per_seg = int(round(fps * segment_seconds))
    if frames_per_seg < 1:
        cap.release()
        raise ValueError("segment_seconds слишком маленький (получилось < 1 кадра)")

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        return []

    src_h, src_w = first_frame.shape[:2]

    # ---- ROI clamp & validate ----
    if roi is None:
        x, y, w, h = 0, 0, src_w, src_h
    else:
        x, y, w, h = roi

        # поджимаем координаты в границы
        x = max(0, min(int(x), src_w - 1))
        y = max(0, min(int(y), src_h - 1))
        w = max(1, int(w))
        h = max(1, int(h))

        # поджимаем ширину/высоту так, чтобы не вылезло за кадр
        if x + w > src_w:
            w = src_w - x
        if y + h > src_h:
            h = src_h - y

    out_w, out_h = w, h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    segments: list[tuple[str, float]] = []

    writer = None
    frame_idx = 0
    seg_idx = 0

    def process_frame(frame):
        if frame is None:
            return None

        if force_resize_to_first_frame:
            if frame.shape[1] != src_w or frame.shape[0] != src_h:
                frame = cv2.resize(frame, (src_w, src_h), interpolation=cv2.INTER_AREA)

        # crop
        frame = frame[y:y + out_h, x:x + out_w]
        return frame

    def open_new_segment(start_frame_idx: int):
        nonlocal writer, seg_idx

        if writer is not None:
            writer.release()

        out_path = os.path.join(out_dir, f"part_{seg_idx:03d}.mp4")

        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Не удалось создать файл: {out_path}")

        start_time_sec = round(start_frame_idx / fps, 1)
        segments.append((out_path, start_time_sec))
        seg_idx += 1

    # кадр 0
    frame0 = process_frame(first_frame)
    if frame0 is None:
        cap.release()
        return []

    open_new_segment(0)
    writer.write(frame0)
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx % frames_per_seg == 0:
            open_new_segment(frame_idx)

        out_frame = process_frame(frame)
        if out_frame is None:
            break

        writer.write(out_frame)
        frame_idx += 1

    if writer is not None:
        writer.release()
    cap.release()

    return segments
