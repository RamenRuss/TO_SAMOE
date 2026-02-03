# models/__init__.py
"""Model backends.

Чтобы добавить новую модель:
1) Создай файл models/my_backend.py
2) Реализуй класс-наследник BaseVideoTextBackend с backend_name = "my"
3) Декорируй его @register_backend
4) Импортируй файл здесь (чтобы регистрация сработала)

Затем можно запускать:
  python main.py --backend my --model-name ...
"""

from .registry import available_backends, create_backend  # noqa: F401

# register built-ins
from .clip_frame_backend import ClipFrameBackend  # noqa: F401
from .xclip_backend import XClipBackend  # noqa: F401
from .dummy_backend import DummyBackend  # noqa: F401
