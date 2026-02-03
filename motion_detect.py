import cv2
import math
from typing import Optional, Tuple


def _clip_roi(roi: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """Обрезает ROI по границам кадра.
    roi: (x, y, w, h)
    """
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        raise ValueError(f"Некорректный ROI (w/h <= 0): {roi}")

    x = max(0, x)
    y = max(0, y)

    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)

    w2 = x2 - x
    h2 = y2 - y
    if w2 <= 0 or h2 <= 0:
        raise ValueError(f"ROI вне кадра после обрезки: {roi} при размере кадра {(frame_w, frame_h)}")

    return x, y, w2, h2


def _apply_roi(frame, roi: Optional[Tuple[int, int, int, int]]):
    if roi is None:
        return frame
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x, y, rw, rh = _clip_roi(roi, w, h)
    return frame[y:y + rh, x:x + rw]


def motion_detect(
    video_path: str,
    min_motion_seconds: float,
    *,
    roi: Optional[Tuple[int, int, int, int]] = None,
    min_area: int = 30,
    threshold_value: int = 13,
    blur_ksize: int = 21,
    max_gap_frames: int = 20
) -> bool:
    """
    Ищет СЕРИЮ длительностью >= min_motion_seconds.

    ROI:
      - если roi задан, анализ движения выполняется ТОЛЬКО внутри roi=(x,y,w,h).
      - ROI НЕ меняет видео на диске (только анализ).
    """

    if min_motion_seconds <= 0:
        raise ValueError("min_motion_seconds должен быть > 0")
    if min_area < 1:
        raise ValueError("min_area должен быть >= 1")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames должен быть >= 0")
    if not (0 <= threshold_value <= 255):
        raise ValueError("threshold_value должен быть в диапазоне 0..255")
    if blur_ksize < 3:
        raise ValueError("blur_ksize должен быть >= 3")
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or not math.isfinite(float(fps)):
        fps = 30.0

    need_frames = int(round(fps * min_motion_seconds))
    if need_frames < 1:
        cap.release()
        raise ValueError("min_motion_seconds слишком маленький (получилось < 1 кадра)")

    def to_gray(frame):
        if frame is None:
            return None
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3:
            ch = frame.shape[2]
            if ch == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ch == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, prev = cap.read()
    if not ret or prev is None:
        cap.release()
        return False

    prev = _apply_roi(prev, roi)
    prev_gray = to_gray(prev)
    if prev_gray is None:
        cap.release()
        return False

    prev_gray = cv2.GaussianBlur(prev_gray, (blur_ksize, blur_ksize), 0)

    in_series = False
    series_len = 0
    gap_run = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = _apply_roi(frame, roi)
        gray = to_gray(frame)
        if gray is None:
            break

        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]

        motion_now = any(cv2.contourArea(c) >= min_area for c in contours)

        if not in_series:
            if motion_now:
                in_series = True
                series_len = 1
                gap_run = 0
        else:
            series_len += 1
            if motion_now:
                gap_run = 0
            else:
                gap_run += 1
                if gap_run > max_gap_frames:
                    in_series = False
                    series_len = 0
                    gap_run = 0

        if in_series and series_len >= need_frames:
            cap.release()
            return True

        prev_gray = gray

    cap.release()
    return False
