from __future__ import annotations
import cv2
import math
import os
from ultralytics import YOLO
from typing import Optional, Tuple, Any, Dict, List
from functools import lru_cache
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dataclasses import dataclass
import numpy as np


def _clip_roi(roi: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> Tuple[int, int, int, int]: #нужна для motion_detect
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


def _apply_roi(frame, roi: Optional[Tuple[int, int, int, int]]): #нужна для motion_detect
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

def person_detect(
    video_path: str,
    n_frames: int,
    model_path: str = "yolo11s.pt",  # ✅ веса YOLOv11 (Ultralytics)
    *,
    sample_every_n_frames: int = 3,    # ✅ ускорение: прогоняем через модель только каждый N-й кадр
    skip_first_n_frames: int = 2,      # ✅ сколько подряд "пустых" проверенных кадров разрешаем внутри серии
    imgsz: int = 416,                  # ✅ размер входа модели: меньше = быстрее, но хуже детект (обычно)
    conf: float = 0.35,                # ✅ порог уверенности: ниже = больше находок, но больше ложных
    iou: float = 0.5,                  # ✅ IoU для NMS: выше = сильнее объединяем близкие боксы
    max_seconds: float | None = None,  # ✅ ограничение анализа по времени (сек)
    device: str | int | None = None,   # ✅ "cpu", 0, "0" (GPU), None = авто
) -> bool:
    """
    Что делает:
    - Читает видео.
    - Запускает детекцию человека (COCO class=0) на выбранных кадрах.
    - Возвращает True, если удалось найти "серию" из n_frames проверенных кадров,
      где человек присутствует, и при этом допускаются "провалы" (кадры без человека)
      длиной до skip_first_n_frames подряд (внутри серии).

    Ключевой момент:
    - "подряд" считается ТОЛЬКО по кадрам, которые реально прогоняются через модель
      (то есть после фильтра sample_every_n_frames).
    """

    # ---------------------------------------------------------------------
    # 1) Валидация входных параметров (чтобы не словить тихие ошибки)
    # ---------------------------------------------------------------------
    if n_frames < 1:
        raise ValueError("n_frames должен быть >= 1")
    if sample_every_n_frames < 1:
        raise ValueError("sample_every_n_frames должен быть >= 1")
    if skip_first_n_frames < 0:
        raise ValueError("skip_first_n_frames должен быть >= 0")

    # ---------------------------------------------------------------------
    # 2) Загрузка модели YOLOv11
    #    Важно: сейчас модель загружается при каждом вызове функции.
    #    Если вы вызываете person_detect много раз, выгоднее:
    #    - загрузить модель один раз снаружи (model = YOLO(...))
    #    - и передавать её в функцию (или сделать отдельную функцию/класс)
    # ---------------------------------------------------------------------
    model = YOLO(model_path)

    # ---------------------------------------------------------------------
    # 3) Открываем видео
    # ---------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    # ---------------------------------------------------------------------
    # 4) Получаем FPS
    #    Это нужно, чтобы корректно посчитать ограничение max_seconds -> max_frames
    # ---------------------------------------------------------------------
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        # Иногда FPS не читается (битый файл/поток/кодек).
        # Тогда берём дефолт, чтобы ограничение по времени всё равно работало.
        fps = 30.0

    # ---------------------------------------------------------------------
    # 5) Если задан лимит по времени, переведём его в лимит по кадрам
    #    frame_idx — индекс кадра в исходном видео (включая пропуски)
    # ---------------------------------------------------------------------
    max_frames = None
    if max_seconds is not None and max_seconds > 0:
        max_frames = int(round(max_seconds * fps))

    frame_idx = 0  # номер текущего кадра во всём видео (все кадры подряд)

    # ---------------------------------------------------------------------
    # 6) Переменные для логики "серии"
    #    run — сколько подряд (в смысле "проверенных") кадров с человеком уже набрали
    #    gap — сколько подряд (в смысле "проверенных") кадров без человека внутри текущей серии
    # ---------------------------------------------------------------------
    run = 0
    gap = 0

    # ---------------------------------------------------------------------
    # 7) Основной цикл чтения кадров
    # ---------------------------------------------------------------------
    try:
        while True:
            ok, frame = cap.read()

            # Если не удалось прочитать кадр — видео закончилось или ошибка чтения
            if not ok or frame is None:
                break

            # Ограничение по длительности анализа (в кадрах исходного видео)
            if max_frames is not None and frame_idx >= max_frames:
                break

            # -------------------------------------------------------------
            # 7.1) Пропуск кадров для ускорения:
            #      например sample_every_n_frames=3 -> берём кадры 0,3,6,9...
            # -------------------------------------------------------------
            if (frame_idx % sample_every_n_frames) != 0:
                frame_idx += 1
                continue

            # -------------------------------------------------------------
            # 7.2) Запускаем детекцию YOLOv11 на выбранном кадре
            #
            # classes=[0]    -> детектим только "person" (в COCO это класс 0)
            # conf/confidence -> отсечение слабых предсказаний
            # iou            -> параметр NMS (слияние/удаление дублей)
            # imgsz          -> размер входа сети (416 быстрее чем 640)
            # device         -> можно принудить CPU/GPU
            #
            # verbose=False  -> чтобы не засорять консоль логами
            # -------------------------------------------------------------
            results = model.predict(
                frame,
                classes=[0],
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )

            # -------------------------------------------------------------
            # 7.3) Извлекаем боксы (bbox)
            # results[0] — т.к. predict возвращает список результатов
            # boxes может быть пустым, если ничего не нашли
            # -------------------------------------------------------------
            boxes = results[0].boxes

            # Если хотя бы один бокс найден — считаем что "человек есть"
            person_now = boxes is not None and len(boxes) > 0

            # -------------------------------------------------------------
            # 7.4) Логика "серии" из n_frames:
            #
            # Если person_now=True:
            #   - увеличиваем run (серия растёт)
            #   - gap сбрасываем (провалов нет)
            #
            # Если person_now=False:
            #   - увеличиваем gap (дырка внутри серии)
            #   - если gap превысил допустимое значение -> серия сбрасывается
            # -------------------------------------------------------------
            if person_now:
                run += 1
                gap = 0
            else:
                gap += 1

                # Слишком длинная "дыра" — считаем серию прерванной
                if gap > skip_first_n_frames:
                    run = 0
                    gap = 0

            # -------------------------------------------------------------
            # 7.5) Условие успеха:
            #      как только набрали n_frames кадров с человеком (с учётом допусков),
            #      возвращаем True
            # -------------------------------------------------------------
            if run >= n_frames:
                return True

            # Увеличиваем индекс исходного кадра
            frame_idx += 1

        # Если дошли до конца видео/лимита — серия так и не набралась
        return False

    finally:
        # Даже если случится исключение — VideoCapture корректно закроется
        cap.release()

def count_video(dir_path: str, recursive: bool = True) -> list[int]:
    """
    Возвращает список индексов [0, 1, 2, ..., N-1],
    где N — количество видеофайлов в папке.

    dir_path: путь к папке
    recursive: True — считать и в подпапках (os.walk), False — только в текущей (os.listdir)
    """
    video_exts = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".m4v")

    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Папка не найдена: {dir_path}")

    count = 0

    if recursive:
        for _, _, files in os.walk(dir_path):
            for name in files:
                if name.lower().endswith(video_exts):
                    count += 1
    else:
        for name in os.listdir(dir_path):
            full_path = os.path.join(dir_path, name)
            if os.path.isfile(full_path) and name.lower().endswith(video_exts):
                count += 1

    return list(range(count))

def delete_video(
    keep_list: list[tuple[str, float]],
    remove_list: list[tuple[str, float]],
    folder_path: str,
    *,
    compare: str = "path",               # "path" или "path_time"
    dry_run: bool = False,               # True = ничего не удаляем (просто пропускаем os.remove)
    allow_outside_folder: bool = False,  # False = запрещаем удалять файлы вне folder_path
) -> None:
    """
    Принимает два массива кортежей (video_path, start_time) и путь к папке.

    Делает разницу: keep_list - remove_list, и удаляет из folder_path файлы,
    которые оказались в этой разнице.

    Ничего не возвращает.
    """

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Папка не найдена: {folder_path}")

    folder_abs = os.path.abspath(folder_path)

    # Нормализация путей для стабильного сравнения (учёт регистра/слешей и т.п.)
    def norm_path(p: str) -> str:
        return os.path.normcase(os.path.abspath(p))

    # Ключ сравнения для элемента (path, time)
    if compare == "path":
        def key(item: tuple[str, float]):
            return norm_path(item[0])
    elif compare == "path_time":
        def key(item: tuple[str, float]):
            return (norm_path(item[0]), round(float(item[1]), 1))
    else:
        raise ValueError('compare должен быть "path" или "path_time"')

    keep_keys = {key(x) for x in keep_list}
    remove_keys = {key(x) for x in remove_list}

    # Разница: есть в keep_list, но нет в remove_list
    diff_keys = keep_keys - remove_keys

    # Из diff_keys восстанавливаем список путей
    for dk in diff_keys:
        # при compare="path_time" dk будет tuple(path, time)
        p_norm = dk[0] if isinstance(dk, tuple) else dk
        p_abs = os.path.abspath(p_norm)

        # Защита: не удалять файлы вне folder_path
        if not allow_outside_folder:
            try:
                if os.path.commonpath([folder_abs, p_abs]) != folder_abs:
                    continue
            except ValueError:
                # например, разные диски на Windows
                continue

        # Удаляем только если это файл
        if os.path.isfile(p_abs):
            if not dry_run:
                os.remove(p_abs)

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

_MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"

@lru_cache(maxsize=1)
def _load_model(device: Optional[str] = None):
    """
    Lazily load tokenizer + model once per process.
    """
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def translate_ru_to_en(text: str, *, max_new_tokens: int = 64) -> str:
    """
    Translate RU -> EN. Intended for short user queries.

    Args:
        text: input Russian text.
        max_new_tokens: generation length cap.

    Returns:
        English translation (best-effort). If translation fails, returns original text.
    """
    s = (text or "").strip()
    if not s:
        return s

    try:
        tokenizer, model, device = _load_model()
        inputs = tokenizer([s], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )
        return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
    except Exception:
        # Safe fallback: do not break retrieval if translation fails.
        return s


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
