from __future__ import annotations

import hashlib
from itertools import islice
from typing import Iterable, Iterator, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def stable_hash_int(value: str, salt: str = "", digest_size: int = 8) -> int:
    payload = f"{salt}::{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=digest_size).digest(), "big")


def l2_normalize(array: np.ndarray, axis: int = 1, eps: float = 1e-6) -> np.ndarray:
    """L2-нормализация, устойчивая к векторам с почти нулевой нормой.

    Если норма меньше ``eps`` (например, текст после нормализации оказался
    пустым и его эмбеддинг получился практически нулевым), вектор возвращается
    как есть — без деления, чтобы не порождать ±inf/NaN, которые ломают
    последующий matmul и hnswlib.
    """
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    safe_norms = np.where(norms < eps, 1.0, norms)
    normalized = array / safe_norms
    # Документы с фактически нулевой нормой остаются нулевыми.
    return np.where(norms < eps, np.zeros_like(array), normalized)


def batched(sequence: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    iterator = iter(sequence)
    while True:
        chunk = list(islice(iterator, batch_size))
        if not chunk:
            break
        yield chunk
