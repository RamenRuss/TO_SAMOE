# config.py
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class RunArgs:
    mode: str = "batch"
    video: str = "video_split"
    query: str = "Человак совершает вандализм"
    show: bool = True


DEFAULT_RUN = RunArgs()


@dataclass
class CachePaths:
    root_dir: str = "./cache"
    index_dir: Optional[str] = None
    results_dir: Optional[str] = None
    run_meta_dir: Optional[str] = None

    index_name_tpl: str = "index_{key}"
    results_name_tpl: str = "results_{key}"

    index_ext: str = ".pt"
    results_ext: str = ".json"

    last_run_name: str = "last_run.json"

    def __post_init__(self) -> None:
        import os

        root = self.root_dir
        if self.index_dir is None:
            self.index_dir = os.path.join(root, "index")
        if self.results_dir is None:
            self.results_dir = os.path.join(root, "results")
        if self.run_meta_dir is None:
            self.run_meta_dir = os.path.join(root, "run_meta")


@dataclass
class MotionDetectConfig:
    enabled: bool = True
    min_motion_seconds: float = 3.0
    min_area: int = 30
    threshold_value: int = 13
    blur_ksize: int = 21
    max_gap_frames: int = 20


@dataclass
class PersonDetectConfig:
    enabled: bool = True
    n_frames: int = 8
    sample_every_n_frames: int = 3
    skip_first_n_frames: int = 2
    model_path: str = "yolo11s.pt"
    imgsz: int = 416
    conf: float = 0.35
    iou: float = 0.5
    max_seconds: Optional[float] = None
    device: str | int | None = None


@dataclass
class PreprocessConfig:
    enabled: bool = True
    input_root_dir: str = "Videos/Vandalism/X" #зедсь нужно что бы путь передавался из temp_path
    splits_root_dir: str = "video_split"
    manifest_name: str = "segments_manifest.json"
    clear_out_dir: bool = True

    segment_seconds: float = 5.0
    min_motion_seconds: float = 3.0
    use_roi: bool = True
    roi: Optional[Tuple[int, int, int, int]] = None

    motion: MotionDetectConfig = field(default_factory=MotionDetectConfig)
    person: PersonDetectConfig = field(default_factory=PersonDetectConfig)


@dataclass
class ModelConfig:
    backend: str = "clip_frame"
    model_name: Optional[str] = None

    device: str = "cpu"
    dtype: str = "fp32"
    use_torch_compile: bool = False

    frame_batch_size: int = 256
    torch_num_threads: Optional[int] = None


@dataclass
class BatchConfig:
    recursive: bool = True

    # Валидационный режим "считать все файлы как одно видео"
    # Реально влияет только на доп. поля в выходе (например, global timeline).
    # Если False — всё равно сортируем все фрагменты в один список, но не строим доп. поля.
    add_global_time_for_validation: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)

    clip_len_frames: int = 8
    clip_stride_frames: int = 64
    sample_strategy: str = "uniform"
    batch_size_clips: int = 32

    normalize_embeddings: bool = True
    fps_hint: Optional[float] = None
    verbose: bool = True

    top_k: int = 5

    batch_out_dir: str = "./batch_out"
    clips_subdir: str = "clips"

    export_clips: bool = False
    export_top_k: int = 5
    pad_sec: float = 2.0
    reencode: bool = True

    cache_index: bool = True
    strict_cache_match: bool = True
    force_reindex: bool = False

    cache_results: bool = True
    use_results_cache: bool = False

    save_top_n_times: int = 5

    paths: CachePaths = field(default_factory=CachePaths)

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

    batch: BatchConfig = field(default_factory=BatchConfig)


def _add_bool_flag(parser: argparse.ArgumentParser, name: str, default: Optional[bool], help_text: str) -> None:
    try:
        action = argparse.BooleanOptionalAction
    except AttributeError:
        action = None

    dest = name.replace("-", "_")

    if action is not None:
        parser.add_argument(f"--{name}", dest=dest, action=action, default=default, help=help_text)
    else:
        parser.add_argument(f"--{name}", dest=dest, action="store_true", default=default, help=help_text)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="VLM", description="Video retrieval: index video -> search by text.")

    p.add_argument("--mode", choices=["single", "batch"], default=None, help="Режим запуска")
    p.add_argument("--video", default=None, help="single: файл; batch: папка")
    p.add_argument("--query", default=None, help="Текстовый запрос")
    _add_bool_flag(p, "show", default=None, help_text="Показывать сегменты (single)")

    _add_bool_flag(p, "preprocess", default=None, help_text="Split+filters перед retrieval (single)")
    p.add_argument("--splits-root-dir", default=None, help="Куда писать нарезку (корень)")
    p.add_argument("--manifest-name", default=None, help="Имя manifest файла в папке нарезки")
    p.add_argument("--segment-seconds", type=float, default=None, help="Длина сегмента при нарезке (сек)")
    _add_bool_flag(p, "use-roi", default=None, help_text="Использовать ROI (требует --roi)")
    p.add_argument("--roi", nargs=4, type=int, default=None, metavar=("X", "Y", "W", "H"), help="ROI как 4 числа: x y w h")

    p.add_argument("--backend", choices=["clip_frame", "xclip"], default=None, help="Бэкенд модели")
    p.add_argument("--model-name", default=None, help="HF model id (зависит от backend)")
    p.add_argument("--device", default=None, help='"cpu" или "cuda:0"')
    p.add_argument("--dtype", choices=["fp16", "fp32"], default=None, help="dtype для модели (cuda: fp16)")
    _add_bool_flag(p, "torch-compile", default=None, help_text="torch.compile(model) (экспериментально)")
    p.add_argument("--frame-batch-size", type=int, default=None, help="Batch size по кадрам (clip_frame)")
    p.add_argument("--torch-num-threads", type=int, default=None, help="torch.set_num_threads(N)")

    p.add_argument("--clip-len", type=int, default=None, help="Длина клипа в кадрах")
    p.add_argument("--clip-stride", type=int, default=None, help="Шаг клипов в кадрах")
    p.add_argument("--batch-size-clips", type=int, default=None, help="Сколько клипов считать за раз")
    p.add_argument("--sample-strategy", choices=["uniform", "head"], default=None)
    _add_bool_flag(p, "normalize-embeddings", default=None, help_text="L2-нормализация эмбеддингов")
    p.add_argument("--fps-hint", type=float, default=None, help="Подсказка FPS если декодер не смог")
    _add_bool_flag(p, "verbose", default=None, help_text="Больше логов")

    p.add_argument("--top-k", type=int, default=None, help="Сколько сегментов вернуть")

    p.add_argument("--batch-out-dir", default=None, help="Куда писать batch результаты")
    p.add_argument("--clips-subdir", default=None, help="Подпапка клипов внутри batch_out_dir")

    _add_bool_flag(p, "export-clips", default=None, help_text="Экспортировать top-k клипы через ffmpeg")
    p.add_argument("--export-top-k", type=int, default=None, help="Сколько клипов экспортировать")
    p.add_argument("--pad-sec", type=float, default=None, help="Запас по краям клипа (сек)")
    _add_bool_flag(p, "reencode", default=None, help_text="reencode (libx264) vs copy")

    _add_bool_flag(p, "cache-index", default=None, help_text="Кэшировать индекс")
    _add_bool_flag(p, "strict-cache-match", default=None, help_text="Сверять cfg_hash")
    _add_bool_flag(p, "force-reindex", default=None, help_text="Игнорировать кэш индекса")

    _add_bool_flag(p, "cache-results", default=None, help_text="Писать results в кэш (json)")
    _add_bool_flag(p, "use-results-cache", default=None, help_text="РАЗРЕШИТЬ брать results из кэша")

    p.add_argument("--save-top-n-times", type=int, default=None, help="Сохранять тайминги top-N (0=выкл)")

    p.add_argument("--cache-dir", default=None, help="Корень кэша (по умолчанию ./cache)")
    p.add_argument("--index-dir", default=None, help="Папка кэша индекса")
    p.add_argument("--results-dir", default=None, help="Папка кэша results")
    p.add_argument("--run-meta-dir", default=None, help="Папка мета")

    return p


def _merge_run_args(ns: argparse.Namespace) -> RunArgs:
    mode = (ns.mode or DEFAULT_RUN.mode).lower()
    video = ns.video or DEFAULT_RUN.video
    query = ns.query or DEFAULT_RUN.query
    show = DEFAULT_RUN.show if ns.show is None else bool(ns.show)
    return RunArgs(mode=mode, video=video, query=query, show=show)


def _apply_cfg_overrides(cfg: Config, ns: argparse.Namespace) -> Config:
    if ns.backend is not None:
        cfg.model.backend = ns.backend
    if ns.model_name is not None:
        cfg.model.model_name = ns.model_name
    if ns.device is not None:
        cfg.model.device = ns.device
    if ns.dtype is not None:
        cfg.model.dtype = ns.dtype
    if ns.torch_compile is not None:
        cfg.model.use_torch_compile = bool(ns.torch_compile)
    if ns.frame_batch_size is not None:
        cfg.model.frame_batch_size = int(ns.frame_batch_size)
    if ns.torch_num_threads is not None:
        cfg.model.torch_num_threads = int(ns.torch_num_threads)

    simple_map = {
        "clip_len": "clip_len_frames",
        "clip_stride": "clip_stride_frames",
        "batch_size_clips": "batch_size_clips",
        "sample_strategy": "sample_strategy",
        "fps_hint": "fps_hint",
        "top_k": "top_k",
        "batch_out_dir": "batch_out_dir",
        "clips_subdir": "clips_subdir",
        "export_top_k": "export_top_k",
        "pad_sec": "pad_sec",
        "save_top_n_times": "save_top_n_times",
    }

    for cli_name, cfg_name in simple_map.items():
        v = getattr(ns, cli_name, None)
        if v is not None:
            setattr(cfg, cfg_name, v)

    bool_map = {
        "normalize_embeddings": "normalize_embeddings",
        "verbose": "verbose",
        "export_clips": "export_clips",
        "reencode": "reencode",
        "cache_index": "cache_index",
        "strict_cache_match": "strict_cache_match",
        "force_reindex": "force_reindex",
        "cache_results": "cache_results",
        "use_results_cache": "use_results_cache",
    }

    for cli_name, cfg_name in bool_map.items():
        v = getattr(ns, cli_name, None)
        if v is not None:
            setattr(cfg, cfg_name, bool(v))

    if ns.cache_dir is not None:
        cfg.paths.root_dir = ns.cache_dir
        cfg.paths.index_dir = None
        cfg.paths.results_dir = None
        cfg.paths.run_meta_dir = None
        cfg.paths.__post_init__()

    if ns.index_dir is not None:
        cfg.paths.index_dir = ns.index_dir
    if ns.results_dir is not None:
        cfg.paths.results_dir = ns.results_dir
    if ns.run_meta_dir is not None:
        cfg.paths.run_meta_dir = ns.run_meta_dir

    if getattr(ns, "preprocess", None) is not None:
        cfg.preprocess.enabled = bool(ns.preprocess)

    if getattr(ns, "splits_root_dir", None) is not None:
        cfg.preprocess.splits_root_dir = ns.splits_root_dir

    if getattr(ns, "manifest_name", None) is not None:
        cfg.preprocess.manifest_name = ns.manifest_name

    if getattr(ns, "segment_seconds", None) is not None:
        cfg.preprocess.segment_seconds = float(ns.segment_seconds)

    if getattr(ns, "use_roi", None) is not None:
        cfg.preprocess.use_roi = bool(ns.use_roi)

    if getattr(ns, "roi", None) is not None:
        r = tuple(int(x) for x in ns.roi) if ns.roi is not None else None
        cfg.preprocess.roi = r

    return cfg


def parse_cli(argv: Optional[Sequence[str]] = None) -> Tuple[RunArgs, Config]:
    parser = build_arg_parser()
    ns = parser.parse_args(argv)

    run = _merge_run_args(ns)
    cfg = _apply_cfg_overrides(Config(), ns)
    return run, cfg
