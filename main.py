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

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file

# Твои модули обработки
from split_video import split_video_web
from motion_detect import motion_detect
from person_detect import person_detect
from count_delete_video import delete_video


# -------------------- НАСТРОЙКИ (можешь менять) --------------------
SEGMENT_SECONDS = 5.0          # длина сегмента (сек)
MIN_MOTION_SECONDS = 3.0       # минимальная "движуха" в сегменте
PERSON_N_FRAMES = 8            # сколько кадров проверять на человека
# ------------------------------------------------------------------


import os, sys

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
#   "timestamps": list[float],  # секунды (старты сегментов которые прошли фильтры)
#   "segments": list[(path, start_sec)],
#   "temp_path": str,           # исходное видео (в temp)
#   "roi": (x,y,w,h) | None,
#   "result": str,
#   "error": str
# }
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()  # чтобы поток обработки и веб-потоки не конфликтовали

# Чтобы пути были короткими — сохраняем сегменты в %TEMP%\dw\<id>\
def _short_job_id(job_id: str) -> str:
    return job_id[:8]


def _job_dir(job_id: str) -> str:
    return os.path.join(tempfile.gettempdir(), "dw", _short_job_id(job_id))


# ---- очистка временных файлов при закрытии приложения ----
_TEMP_FILES: set[str] = set()
_TEMP_DIRS: set[str] = set()


def cleanup_temp() -> None:
    # Удаляем загруженные видео
    for p in list(_TEMP_FILES):
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass

    # Удаляем папки с сегментами
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


def process_job(job_id: str) -> None:
    """
    Фоновая обработка:
    1) нарезка (split_video_web) -> all_segments: [(path, start_sec)]
    2) поштучная проверка каждого сегмента (motion -> person)
       если сегмент подходит — сразу добавляем start_sec в timestamps (чтобы UI показывал постепенно)
    3) удаление ненужных сегментов delete_video(all, kept)
    4) статус done/error
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return

    try:
        src_video = job["temp_path"]
        roi = job.get("roi")  # (x,y,w,h) в пикселях или None

        out_dir = _job_dir(job_id)
        os.makedirs(out_dir, exist_ok=True)
        _TEMP_DIRS.add(out_dir)

        # 1) Нарезка видео (ROI учитывается внутри split_video_web)
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

        # 2) Поштучная фильтрация + постепенное добавление таймкодов
        for i, (seg_path, start_sec) in enumerate(all_segments, start=1):
            ok_motion = motion_detect(seg_path, MIN_MOTION_SECONDS)
            ok_person = False
            if ok_motion:
                ok_person = person_detect(seg_path, PERSON_N_FRAMES, weights_path)

            if ok_motion and ok_person:
                kept.append((seg_path, start_sec))
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                    if job:
                        job["timestamps"].append(float(start_sec))  # UI сразу увидит новый таймкод

            # обновляем прогресс
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job:
                    job["progress"] = int(i * 100 / total)

        # 3) Удаляем лишние сегменты, оставляем только kept
        delete_video(all_segments, kept, folder_path=out_dir)

        # 4) Финализируем job
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job["segments"] = kept
                job["timestamps"] = sorted([float(t) for t in job["timestamps"]])
                job["result"] = f"Готово. Осталось сегментов: {len(kept)}"
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
            "result": "",
            "error": "",
        }

    return redirect(url_for("roi", job_id=job_id))


@app.get("/roi/<job_id>")
def roi(job_id: str):
    """Страница выбора ROI."""
    with JOBS_LOCK:
        if job_id not in JOBS:
            return "Задача не найдена", 404
    return render_template("roi.html", job_id=job_id)


@app.post("/roi/<job_id>")
def roi_submit(job_id: str):
    """
    Получаем ROI (x,y,w,h) из формы и запускаем обработку в фоне.
    Сразу кидаем пользователя на /result/<job_id> (плеер),
    где таймкоды будут появляться постепенно.
    """
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

    # ВАЖНО: сразу на плеер/результаты
    return redirect(url_for("result", job_id=job_id))


@app.get("/result/<job_id>")
def result(job_id: str):
    """Плеер/страница результатов. Может быть processing/done/error."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return "Задача не найдена", 404
    return render_template("result.html", job=job, job_id=job_id)


@app.get("/api/job/<job_id>")
def api_job(job_id: str):
    """API для фронта: статус, прогресс и текущие таймкоды (постепенно пополняются)."""
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
    """Отдаём исходное видео для просмотра на roi/result страницах."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return "Видео не найдено", 404

    path = job.get("temp_path")
    if not path or not os.path.isfile(path):
        return "Видео не найдено", 404

    mime, _ = mimetypes.guess_type(path)
    return send_file(path, mimetype=mime or "video/mp4", conditional=True)


# (опционально) старый /processing и /api/status можно оставить, но сейчас не нужен.
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
