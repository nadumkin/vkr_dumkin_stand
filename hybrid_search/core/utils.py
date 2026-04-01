from __future__ import annotations

import hashlib
from itertools import islice
from typing import Iterable, Iterator, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def stable_hash_int(value: str, salt: str = "", digest_size: int = 8) -> int:
    payload = f"{salt}::{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=digest_size).digest(), "big")


def l2_normalize(array: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.clip(norms, eps, None)


def batched(sequence: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    iterator = iter(sequence)
    while True:
        chunk = list(islice(iterator, batch_size))
        if not chunk:
            break
        yield chunk
