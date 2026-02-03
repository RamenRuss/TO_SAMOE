# main_vlm.py
# Web (Flask) entrypoint + VLM pipeline.
#
# ROI logic is the same as in main.py:
# - ROI is applied during splitting (split_video_web crops segments)
# - motion/person detectors run WITHOUT roi (they see already-cropped segments)

from __future__ import annotations

import os
import sys
import time
import uuid
import socket
import shutil
import threading
import webbrowser
import tempfile
import atexit
import signal
import mimetypes
import json
import re
import warnings
from dataclasses import asdict

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file

# ---------- preprocessing (как в main.py) ----------
from split_video import split_video_web
from motion_detect import motion_detect
from person_detect import person_detect
from count_delete_video import delete_video

# ---------- VLM ----------
from translate import translate_ru_to_en
from cache_io import ensure_all_cache_dirs
from config import Config
from models import create_backend
from run_one_video import process_one_video

# Disable HuggingFace symlinks warning on Windows (PyInstaller)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message="Recommended: pip install sacremoses.*")
warnings.filterwarnings("ignore", message="Xet Storage is enabled for this repo.*")

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


def has_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text or ""))


# -------------------- НАСТРОЙКИ (можешь менять) --------------------
SEGMENT_SECONDS = 5.0          # длина сегмента (сек)
MIN_MOTION_SECONDS = 3.0       # минимальная "движуха" в сегменте
PERSON_N_FRAMES = 8            # сколько кадров проверять на человека
# ------------------------------------------------------------------


def resource_path(filename: str) -> str:
    # 1) onefile: внутри временной папки PyInstaller
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        p = os.path.join(sys._MEIPASS, filename)
        if os.path.exists(p):
            return p

    # 2) рядом с exe (dist)
    exe_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else None
    if exe_dir:
        p = os.path.join(exe_dir, filename)
        if os.path.exists(p):
            return p

    # 3) рядом с .py (режим разработки)
    return os.path.join(os.path.dirname(__file__), filename)


weights_path = resource_path("yolo11s.pt")


def is_frozen() -> bool:
    """True если запущено как PyInstaller exe."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def runtime_dir() -> str:
    """
    Папка запуска:
    - .py: рядом с файлом
    - .exe: рядом с exe
    """
    return os.path.dirname(sys.executable) if is_frozen() else os.path.dirname(os.path.abspath(__file__))


def resource_dir(name: str) -> str:
    """
    Где искать templates/static:
    1) рядом с exe/скриптом (удобно менять без пересборки)
    2) внутри PyInstaller onefile (_MEIPASS)
    """
    local = os.path.join(runtime_dir(), name)
    if os.path.isdir(local):
        return local
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def find_free_port(host: str = "127.0.0.1") -> int:
    """Берём свободный порт, чтобы не конфликтовать с другими приложениями."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


app = Flask(
    __name__,
    template_folder=resource_dir("templates"),
    static_folder=resource_dir("static"),
)

# JOBS хранит состояние обработки
# job_id -> {
#   "status": "await_roi" | "processing" | "done" | "error",
#   "progress": int(0..100),
#   "timestamps": list[float],  # секунды (старты VLM-top сегментов)
#   "segments": list[(path, start_sec)],
#   "temp_path": str,           # исходное видео (в temp)
#   "roi": (x,y,w,h) | None,
#   "query_text": str,
#   "result": str,
#   "error": str
# }
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

# Чтобы пути были короткими — сохраняем сегменты в %TEMP%\dw\<id>\
def _short_job_id(job_id: str) -> str:
    return job_id[:8]


def _job_dir(job_id: str) -> str:
    return os.path.join(tempfile.gettempdir(), "dw", _short_job_id(job_id))


# ---- очистка временных файлов при закрытии приложения ----
_TEMP_FILES: set[str] = set()
_TEMP_DIRS: set[str] = set()


def cleanup_temp() -> None:
    for p in list(_TEMP_FILES):
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass

    for d in list(_TEMP_DIRS):
        try:
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


atexit.register(cleanup_temp)


def _handle_exit(signum, frame):
    cleanup_temp()
    raise SystemExit


signal.signal(signal.SIGINT, _handle_exit)
try:
    signal.signal(signal.SIGTERM, _handle_exit)
except Exception:
    pass
# ----------------------------------------------------------


# -------- backend cache (чтобы не грузить модель каждый раз) --------
_BACKEND_LOCK = threading.Lock()
_BACKEND = None
_BACKEND_CFG_KEY = None


def _get_backend(cfg: Config):
    """
    Lazy-load backend and reuse between jobs.
    If model config changes (backend/model_name/device/dtype), reload.
    """
    global _BACKEND, _BACKEND_CFG_KEY
    key = (
        getattr(cfg.model, "backend", None),
        getattr(cfg.model, "model_name", None),
        getattr(cfg.model, "device", None),
        getattr(cfg.model, "dtype", None),
        getattr(cfg.model, "use_torch_compile", None),
    )
    with _BACKEND_LOCK:
        if _BACKEND is None or _BACKEND_CFG_KEY != key:
            _BACKEND = create_backend(cfg, load=True)
            _BACKEND_CFG_KEY = key
        return _BACKEND
# -------------------------------------------------------------------


def _make_manifest(
    *,
    source_video: str,
    out_dir: str,
    segment_seconds: float,
    roi: tuple[int, int, int, int] | None,
    kept: list[tuple[str, float]],
    manifest_name: str = "segments_manifest.json",
) -> str:
    manifest_path = os.path.join(out_dir, manifest_name)
    manifest = {
        "source_video": os.path.abspath(source_video),
        "segments_dir": os.path.abspath(out_dir),
        "segment_seconds": float(segment_seconds),
        "use_roi": bool(roi is not None),
        "roi": list(roi) if roi is not None else None,
        "segments": [
            {"path": os.path.abspath(p), "start_time_sec": float(s)}
            for (p, s) in kept
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def process_job(job_id: str) -> None:
    """
    Фоновая обработка (WEB + VLM):

    1) split_video_web(input, roi=roi) -> all_segments: [(path, start_sec)]
       (ROI логика как в main.py: split режет и кропает сегменты)
    2) motion_detect + person_detect на сегментах
    3) delete_video(all, kept)
    4) manifest.json для kept
    5) VLM retrieve top-k по query_text через process_one_video(... presegmented_manifest_path=manifest)
    6) job["timestamps"] = start_time_sec top-сегментов
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return

    try:
        src_video = job["temp_path"]
        roi = job.get("roi")
        query_text = (job.get("query_text") or "").strip()

        if not query_text:
            raise ValueError("Пустой текстовый запрос (query_text).")

        # RU -> EN (как в VLM cli)
        if has_cyrillic(query_text):
            q2 = translate_ru_to_en(query_text)
            if q2:
                query_text = q2

        out_dir = _job_dir(job_id)
        os.makedirs(out_dir, exist_ok=True)
        _TEMP_DIRS.add(out_dir)

        # 1) split (ROI как в main.py)
        all_segments = split_video_web(
            input_path=src_video,
            out_dir=out_dir,
            segment_seconds=SEGMENT_SECONDS,
            roi=roi,
        )

        if not all_segments:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job:
                    job["progress"] = 100
                    job["timestamps"] = []
                    job["segments"] = []
                    job["result"] = "Сегменты не получились (пустое/битое видео?)"
                    job["status"] = "done"
            return

        total = len(all_segments)
        kept: list[tuple[str, float]] = []

        # 2) фильтры (ROI НЕ передаём, т.к. сегменты уже кропнуты при split)
        for i, (seg_path, start_sec) in enumerate(all_segments, start=1):
            ok_motion = motion_detect(seg_path, MIN_MOTION_SECONDS)
            ok_person = False
            if ok_motion:
                ok_person = person_detect(seg_path, PERSON_N_FRAMES, weights_path)

            if ok_motion and ok_person:
                kept.append((seg_path, start_sec))

            # прогресс 0..60%
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job:
                    job["progress"] = int(i * 60 / total)

        # 3) чистим
        delete_video(all_segments, kept, folder_path=out_dir)

        # 4) manifest
        manifest_path = _make_manifest(
            source_video=src_video,
            out_dir=out_dir,
            segment_seconds=SEGMENT_SECONDS,
            roi=roi,
            kept=kept,
        )

        # 5) VLM retrieve
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job["progress"] = 70

        cfg = Config()
        # cache рядом с запуском (чтобы в exe было удобно)
        cfg.paths.root_dir = os.path.join(runtime_dir(), "cache")
        ensure_all_cache_dirs(cfg=cfg)

        backend = _get_backend(cfg)

        out = process_one_video(
            video_path=src_video,
            query=query_text,
            backend=backend,
            cfg=cfg,
            cfg_full=asdict(cfg),
            presegmented_manifest_path=manifest_path,
        )

        results_list = out.get("results_list") or []
        timestamps = []
        for r in results_list:
            t = r.get("start_time_sec")
            if t is None:
                continue
            timestamps.append(float(t))

        # 6) финал
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job["segments"] = kept
                job["timestamps"] = timestamps
                job["result"] = f"Готово. Осталось сегментов: {len(kept)}. VLM top: {len(timestamps)}"
                job["status"] = "done"
                job["progress"] = 100

    except Exception as e:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job["status"] = "error"
                job["error"] = str(e)


# -------------------- ROUTES --------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/run")
def run():
    """
    1) Получаем видео из формы
    2) Сохраняем во временный файл (короткий путь)
    3) Создаём job и отправляем на страницу выбора ROI
    """
    file = request.files.get("video_file")
    if not file or not file.filename:
        return "Файл не выбран", 400

    query_text = (request.form.get("query_text") or "").strip()

    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    fd, temp_path = tempfile.mkstemp(prefix="dw_", suffix=ext)
    os.close(fd)
    file.save(temp_path)
    _TEMP_FILES.add(temp_path)

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "await_roi",
            "progress": 0,
            "timestamps": [],
            "segments": [],
            "temp_path": temp_path,
            "roi": None,
            "query_text": query_text,
            "result": "",
            "error": "",
        }

    return redirect(url_for("roi", job_id=job_id))


@app.get("/roi/<job_id>")
def roi(job_id: str):
    with JOBS_LOCK:
        if job_id not in JOBS:
            return "Задача не найдена", 404
    return render_template("roi.html", job_id=job_id)


@app.post("/roi/<job_id>")
def roi_submit(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return "Задача не найдена", 404

    try:
        x = int(float(request.form.get("x", "0")))
        y = int(float(request.form.get("y", "0")))
        w = int(float(request.form.get("w", "0")))
        h = int(float(request.form.get("h", "0")))
    except ValueError:
        return "ROI некорректен", 400

    if w <= 0 or h <= 0:
        return "ROI некорректен", 400

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return "Задача не найдена", 404
        job["roi"] = (x, y, w, h)
        job["status"] = "processing"
        job["progress"] = 0
        job["timestamps"] = []
        job["segments"] = []
        job["result"] = ""
        job["error"] = ""

    threading.Thread(target=process_job, args=(job_id,), daemon=True).start()
    return redirect(url_for("result", job_id=job_id))


@app.get("/result/<job_id>")
def result(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return "Задача не найдена", 404
    return render_template("result.html", job=job, job_id=job_id)


@app.get("/api/job/<job_id>")
def api_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"status": "not_found"}), 404

        return jsonify({
            "status": job.get("status", "unknown"),
            "progress": int(job.get("progress", 0)),
            "timestamps": job.get("timestamps", []),
            "result": job.get("result", ""),
            "error": job.get("error", ""),
        })


@app.get("/video/<job_id>")
def video(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return "Видео не найдено", 404

    path = job.get("temp_path")
    if not path or not os.path.isfile(path):
        return "Видео не найдено", 404

    mime, _ = mimetypes.guess_type(path)
    return send_file(path, mimetype=mime or "video/mp4", conditional=True)


@app.get("/processing/<job_id>")
def processing(job_id: str):
    with JOBS_LOCK:
        if job_id not in JOBS:
            return "Задача не найдена", 404
    return render_template("processing.html", job_id=job_id)


@app.get("/api/status/<job_id>")
def api_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"status": "not_found"}), 404
        return jsonify({"status": job.get("status", "unknown")})


# -------------------- RUN --------------------

def open_browser_later(url: str) -> None:
    time.sleep(0.7)
    webbrowser.open(url, new=1)


def main():
    host = "127.0.0.1"
    port = find_free_port(host)
    url = f"http://{host}:{port}/"

    threading.Thread(target=open_browser_later, args=(url,), daemon=True).start()
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
