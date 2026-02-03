// static/app.js

document.addEventListener("DOMContentLoaded", () => {
  initFilePickerUI();
  initRoiPage();          // если открыта roi.html
  initResultPageLive();   // если открыта result.html
});

/* -------------------- Главная: выбор файла -------------------- */
function initFilePickerUI() {
  const input = document.getElementById("video_file");
  const nameEl = document.getElementById("file_name");
  const btn = document.getElementById("file_btn");
  if (!input || !nameEl || !btn) return;

  const update = () => {
    const f = input.files && input.files[0];
    if (f) {
      nameEl.textContent = f.name;
      btn.textContent = "Изменить файл";
      btn.classList.add("is-selected");
    } else {
      nameEl.textContent = "Файл не выбран";
      btn.textContent = "Выбрать файл";
      btn.classList.remove("is-selected");
    }
  };

  input.addEventListener("change", update);
  update();
}

/* -------------------- Helpers -------------------- */
function clamp(n, a, b) {
  return Math.min(b, Math.max(a, n));
}

function formatTime(sec) {
  sec = Math.max(0, Number(sec) || 0);
  const s = Math.floor(sec % 60);
  const m = Math.floor((sec / 60) % 60);
  const h = Math.floor(sec / 3600);

  const ss = String(s).padStart(2, "0");
  const mm = h > 0 ? String(m).padStart(2, "0") : String(m);
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function normalizeMarkers(arr) {
  return (arr || [])
    .map((x) => Number(x))
    .filter((x) => Number.isFinite(x) && x >= 0)
    .sort((a, b) => a - b);
}

function markersEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/* -------------------- RESULT: live markers + progress -------------------- */
function initResultPageLive() {
  const page = document.querySelector("[data-page='result']");
  if (!page) return;

  const jobId = page.getAttribute("data-job-id");
  if (!jobId) return;

  const statusTextEl = document.getElementById("status_text");
  const statusSubEl = document.getElementById("status_subtext");
  const progressEl = document.getElementById("progress_text");

  const listEl = document.getElementById("tc_list");
  const video = document.getElementById("result_video");
  const seek = document.getElementById("video_seek");
  const curEl = document.getElementById("cur_time");
  const durEl = document.getElementById("dur_time");
  const markerLayer = document.getElementById("timeline_markers");

  if (!listEl || !video || !seek || !curEl || !durEl || !markerLayer) return;

  let markers = [];
  try {
    markers = JSON.parse(page.getAttribute("data-markers") || "[]");
  } catch {
    markers = [];
  }
  markers = normalizeMarkers(markers);

  function jumpTo(t) {
    if (!Number.isFinite(t)) return;
    const dur = video.duration || 0;
    if (dur > 0) video.currentTime = clamp(t, 0, dur);
    else video.currentTime = Math.max(0, t);
    video.play?.();
  }

  function renderList() {
    listEl.innerHTML = "";

    if (markers.length === 0) {
      const li = document.createElement("li");
      li.className = "tc-item";
      li.textContent = "Пока нет таймкодов…";
      listEl.appendChild(li);
      return;
    }

    for (const t of markers) {
      const li = document.createElement("li");
      li.className = "tc-item";

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "tc-btn";
      btn.addEventListener("click", () => jumpTo(t));

      const timeSpan = document.createElement("span");
      timeSpan.className = "tc-time";
      timeSpan.textContent = formatTime(t);

      const hintSpan = document.createElement("span");
      hintSpan.className = "tc-jump";
      hintSpan.textContent = "перейти ↗";

      btn.appendChild(timeSpan);
      btn.appendChild(hintSpan);
      li.appendChild(btn);
      listEl.appendChild(li);
    }
  }

  function renderMarkers() {
    const dur = video.duration || 0;
    markerLayer.innerHTML = "";

    if (!dur || markers.length === 0) return;

    for (const t of markers) {
      if (!Number.isFinite(t) || t < 0) continue;

      let x = (t / dur) * 100;
      x = Math.max(0.8, Math.min(99.2, x)); // чтобы точки не вылезали за края

      const dot = document.createElement("div");
      dot.className = "timeline-marker";
      dot.style.left = `${x}%`;
      dot.title = formatTime(t);
      dot.addEventListener("click", () => jumpTo(t));
      markerLayer.appendChild(dot);
    }
  }

  function syncUI() {
    const dur = video.duration || 0;
    const cur = video.currentTime || 0;

    durEl.textContent = formatTime(dur);
    curEl.textContent = formatTime(cur);

    if (dur > 0) {
      seek.value = String(Math.round((cur / dur) * 1000));
    } else {
      seek.value = "0";
    }
  }

  video.addEventListener("loadedmetadata", () => {
    renderMarkers();
    syncUI();
  });
  video.addEventListener("timeupdate", syncUI);

  seek.addEventListener("input", () => {
    const dur = video.duration || 0;
    if (!dur) return;
    const v = clamp(Number(seek.value) || 0, 0, 1000);
    video.currentTime = (v / 1000) * dur;
  });

  // первичный рендер
  renderList();
  if (video.readyState >= 1) {
    renderMarkers();
    syncUI();
  }

  // live polling /api/job/<id>
  let stopped = false;

  async function tick() {
    if (stopped) return;

    try {
      const r = await fetch(`/api/job/${jobId}`, { cache: "no-store" });
      if (!r.ok) throw new Error("bad response");
      const data = await r.json();

      const st = data.status || "processing";
      const prog = Number(data.progress);
      const nextMarkers = normalizeMarkers(data.timestamps || []);

      // статусные тексты
      if (st === "error") {
        if (statusTextEl) statusTextEl.textContent = "Ошибка ❌";
        if (statusSubEl) statusSubEl.textContent = data.error || "Что-то пошло не так.";
        if (progressEl) progressEl.textContent = "";
        stopped = true;
        return;
      }

      if (st === "done") {
        if (statusTextEl) statusTextEl.textContent = "Обработка закончена ✅";
        if (statusSubEl) statusSubEl.textContent = "Таймкоды отмечены на таймлайне. Можно переходить по списку.";
        if (progressEl) progressEl.textContent = "100%";
        stopped = true;
      } else {
        if (statusTextEl) statusTextEl.textContent = "Идёт обработка…";
        if (statusSubEl) statusSubEl.textContent = "Первые таймкоды появятся по мере готовности.";
        if (progressEl) progressEl.textContent = `${Number.isFinite(prog) ? prog : 0}%`;
      }

      // таймкоды (постепенно)
      if (!markersEqual(markers, nextMarkers)) {
        markers = nextMarkers;
        renderList();
        if (video.duration) renderMarkers();
      }

      if (!stopped) setTimeout(tick, 800);
    } catch (e) {
      // тихо повторяем
      if (!stopped) setTimeout(tick, 1000);
    }
  }

  tick();
}

/* -------------------- ROI: move + resize + dim -------------------- */
function initRoiPage() {
  const page = document.querySelector("[data-page='roi']");
  if (!page) return;

  const video = document.getElementById("roi_video");
  const box = document.getElementById("roi_box");
  const wrap = document.getElementById("roi_wrap");

  const inX = document.getElementById("roi_x");
  const inY = document.getElementById("roi_y");
  const inW = document.getElementById("roi_w");
  const inH = document.getElementById("roi_h");

  const dimTop = document.querySelector(".roi-dim-top");
  const dimLeft = document.querySelector(".roi-dim-left");
  const dimRight = document.querySelector(".roi-dim-right");
  const dimBottom = document.querySelector(".roi-dim-bottom");

  if (!video || !box || !wrap || !inX || !inY || !inW || !inH) return;

  const MIN_W = 40;
  const MIN_H = 40;

  let mode = null;      // "move" | "resize"
  let handle = null;    // "nw","n","ne","w","e","sw","s","se"
  let startMx = 0, startMy = 0;
  let start = { left: 0, top: 0, width: 0, height: 0 };

  function getWrapRect() {
    return video.getBoundingClientRect();
  }

  function updateDim() {
    if (!dimTop || !dimLeft || !dimRight || !dimBottom) return;

    const r = getWrapRect();
    const b = box.getBoundingClientRect();

    const left = b.left - r.left;
    const top = b.top - r.top;
    const w = b.width;
    const h = b.height;

    dimTop.style.left = "0px";
    dimTop.style.top = "0px";
    dimTop.style.width = `${r.width}px`;
    dimTop.style.height = `${top}px`;

    dimLeft.style.left = "0px";
    dimLeft.style.top = `${top}px`;
    dimLeft.style.width = `${left}px`;
    dimLeft.style.height = `${h}px`;

    dimRight.style.left = `${left + w}px`;
    dimRight.style.top = `${top}px`;
    dimRight.style.width = `${Math.max(0, r.width - (left + w))}px`;
    dimRight.style.height = `${h}px`;

    dimBottom.style.left = "0px";
    dimBottom.style.top = `${top + h}px`;
    dimBottom.style.width = `${r.width}px`;
    dimBottom.style.height = `${Math.max(0, r.height - (top + h))}px`;
  }

  function updateHiddenInputs() {
    const r = getWrapRect();
    const b = box.getBoundingClientRect();

    const relX = (b.left - r.left) / r.width;
    const relY = (b.top - r.top) / r.height;
    const relW = b.width / r.width;
    const relH = b.height / r.height;

    const vw = video.videoWidth || 0;
    const vh = video.videoHeight || 0;

    inX.value = String(Math.max(0, Math.round(relX * vw)));
    inY.value = String(Math.max(0, Math.round(relY * vh)));
    inW.value = String(Math.max(1, Math.round(relW * vw)));
    inH.value = String(Math.max(1, Math.round(relH * vh)));
  }

  function setBoxRect(left, top, width, height) {
    const r = getWrapRect();

    width = clamp(width, MIN_W, r.width);
    height = clamp(height, MIN_H, r.height);

    left = clamp(left, 0, r.width - width);
    top = clamp(top, 0, r.height - height);

    box.style.left = `${left}px`;
    box.style.top = `${top}px`;
    box.style.width = `${width}px`;
    box.style.height = `${height}px`;

    updateDim();
    updateHiddenInputs();
  }

  function beginInteraction(e, newMode, newHandle) {
    e.preventDefault();
    mode = newMode;
    handle = newHandle || null;

    startMx = e.clientX;
    startMy = e.clientY;

    start.left = box.offsetLeft;
    start.top = box.offsetTop;
    start.width = box.offsetWidth;
    start.height = box.offsetHeight;

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", endInteraction, { once: true });
  }

  function endInteraction() {
    mode = null;
    handle = null;
    window.removeEventListener("mousemove", onMove);
  }

  function onMove(e) {
    const r = getWrapRect();
    const dx = e.clientX - startMx;
    const dy = e.clientY - startMy;

    if (mode === "move") {
      setBoxRect(start.left + dx, start.top + dy, start.width, start.height);
      return;
    }

    if (mode === "resize") {
      let left = start.left;
      let top = start.top;
      let width = start.width;
      let height = start.height;

      const right = start.left + start.width;
      const bottom = start.top + start.height;

      // X
      if (handle.includes("w")) {
        left = clamp(start.left + dx, 0, right - MIN_W);
        width = right - left;
      }
      if (handle.includes("e")) {
        const newRight = clamp(right + dx, left + MIN_W, r.width);
        width = newRight - left;
      }

      // Y
      if (handle.includes("n")) {
        top = clamp(start.top + dy, 0, bottom - MIN_H);
        height = bottom - top;
      }
      if (handle.includes("s")) {
        const newBottom = clamp(bottom + dy, top + MIN_H, r.height);
        height = newBottom - top;
      }

      setBoxRect(left, top, width, height);
    }
  }

  // init
  video.addEventListener("loadedmetadata", () => {
    const r = getWrapRect();
    setBoxRect(
      Math.round(r.width * 0.10),
      Math.round(r.height * 0.10),
      Math.round(r.width * 0.55),
      Math.round(r.height * 0.55)
    );
  });

  // move by box
  box.addEventListener("mousedown", (e) => {
    if (e.target && e.target.classList.contains("roi-handle")) return;
    beginInteraction(e, "move", null);
  });

  // resize by handles
  box.querySelectorAll(".roi-handle").forEach((h) => {
    h.addEventListener("mousedown", (e) => {
      const key = e.target.getAttribute("data-h");
      beginInteraction(e, "resize", key);
    });
  });

  window.addEventListener("resize", () => {
    updateDim();
    updateHiddenInputs();
  });
}
