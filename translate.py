"""
translate.py
Offline (open-source) RU->EN translator for user queries.

Uses Helsinki-NLP/opus-mt-ru-en (OPUS-MT) via Hugging Face Transformers.
Model license: CC-BY 4.0 (see ReadMe.md for attribution).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
