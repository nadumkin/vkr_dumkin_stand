"""Классические бейслайны компрессии векторных представлений.

Реализации:
    - ``LSHHasher``  — случайное гауссово гиперплоскостное хеширование
                       (cosine-LSH / SimHash). Бейслайн без обучения.
    - ``ITQHasher``  — Iterative Quantization [Gong et al., IEEE TPAMI 2013].
                       Учит ортогональную ротацию R, минимизирующую
                       ошибку квантизации в бинарный гиперкуб.
    - ``ProductQuantizer`` — Product Quantization [Jégou et al., IEEE TPAMI 2011].
                       Разбивает пространство на M подпространств и квантует
                       каждое с помощью K-means на K центроидов.
                       Поиск выполняется через ADC (Asymmetric Distance Computation).

Все три метода работают с numpy-массивами и не зависят от torch — это упрощает
их сравнение с обучаемой нейронной хэш-головой.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# LSH — случайные гауссовы гиперплоскости
# ---------------------------------------------------------------------------
class LSHHasher:
    """Random hyperplane LSH (cosine-LSH / SimHash).

    Для каждого бита используется случайная гиперплоскость через начало
    координат с гауссовым нормалем. Бит вектора — знак скалярного произведения
    с этой нормалью. Без обучения; коды детерминированы по seed.

    При size кода равном code_bits и длине вектора d, сложность кодирования
    одного вектора O(d × code_bits), памяти под гиперплоскости O(d × code_bits).
    """

    def __init__(self, input_dim: int, code_bits: int, seed: int = 17) -> None:
        self.input_dim = int(input_dim)
        self.code_bits = int(code_bits)
        self.seed = int(seed)
        rng = np.random.RandomState(self.seed)
        self.hyperplanes = rng.randn(self.input_dim, self.code_bits).astype(np.float32)
        # Время операций (заполняется в fit/encode)
        self.last_fit_time_ms: float = 0.0

    def fit(self, embeddings: np.ndarray) -> "LSHHasher":
        """LSH не обучается, но метод оставлен для единообразия интерфейса."""
        self.last_fit_time_ms = 0.0
        return self

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """Возвращает коды формы (N, code_bits) в {-1, +1}."""
        scores = embeddings.astype(np.float32, copy=False) @ self.hyperplanes
        return np.where(scores > 0, 1, -1).astype(np.int8)


# ---------------------------------------------------------------------------
# ITQ — Iterative Quantization
# ---------------------------------------------------------------------------
class ITQHasher:
    """Iterative Quantization (Gong et al., 2013).

    Алгоритм:
        1. Центрируем данные.
        2. PCA-проекция в пространство размерности code_bits.
        3. Случайная ортогональная инициализация R.
        4. Повторяем n_iter раз:
            a) B = sign(X @ R)
            b) Procrustes решение: R = V U^T, где U S V^T = SVD(X^T B).

    Минимизирует ‖sign(XR) − XR‖² при условии R^T R = I.
    """

    def __init__(
        self,
        input_dim: int,
        code_bits: int,
        n_iter: int = 50,
        seed: int = 17,
    ) -> None:
        self.input_dim = int(input_dim)
        self.code_bits = int(code_bits)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.mean: Optional[np.ndarray] = None
        self.pca: Optional[np.ndarray] = None      # (input_dim, code_bits)
        self.rotation: Optional[np.ndarray] = None  # (code_bits, code_bits)
        self.last_fit_time_ms: float = 0.0

    def fit(self, embeddings: np.ndarray) -> "ITQHasher":
        started = time.perf_counter()
        X = embeddings.astype(np.float32, copy=False)
        self.mean = X.mean(axis=0, keepdims=True).astype(np.float32)
        Xc = X - self.mean

        # 1. PCA через SVD: top code_bits главных компонент.
        if self.code_bits > self.input_dim:
            # Теоретически возможно, но бессмысленно.
            self.pca = np.eye(self.input_dim, self.code_bits, dtype=np.float32)
        else:
            # np.linalg.svd возвращает Vh = V^T размерности (min(N,d), d)
            _, _, Vh = np.linalg.svd(Xc, full_matrices=False)
            self.pca = Vh[: self.code_bits].T.astype(np.float32, copy=False)

        Xp = Xc @ self.pca  # (N, code_bits)

        # 2. Случайная ортогональная инициализация R через QR/SVD.
        rng = np.random.RandomState(self.seed)
        R0 = rng.randn(self.code_bits, self.code_bits).astype(np.float32)
        U_init, _, Vh_init = np.linalg.svd(R0)
        R = (U_init @ Vh_init).astype(np.float32)

        # 3. Итеративное обновление по Procrustes.
        for _ in range(self.n_iter):
            B = np.sign(Xp @ R)
            B[B == 0] = 1.0
            U, _, Vh = np.linalg.svd(Xp.T @ B)
            # argmin ||B - X R||^2 при ортогональном R даёт R = V U^T
            R = (Vh.T @ U.T).astype(np.float32)

        self.rotation = R
        self.last_fit_time_ms = (time.perf_counter() - started) * 1000.0
        return self

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        if self.mean is None or self.pca is None or self.rotation is None:
            raise RuntimeError("ITQHasher.fit() должен быть вызван до encode().")
        X = embeddings.astype(np.float32, copy=False) - self.mean
        Xp = X @ self.pca @ self.rotation
        codes = np.where(Xp > 0, 1, -1).astype(np.int8)
        return codes


# ---------------------------------------------------------------------------
# PQ — Product Quantization
# ---------------------------------------------------------------------------
@dataclass
class PQSearchResult:
    indices: np.ndarray   # (k,) — индексы документов
    distances: np.ndarray # (k,) — приближённые квадратные L2 расстояния


class ProductQuantizer:
    """Product Quantization (Jégou et al., 2011).

    Делит вектор размерности D на M подпространств размерности D/M, на каждом
    обучает кодовую книгу из K центроидов через K-means. Каждый вектор
    кодируется M индексами (целыми от 0 до K−1). Суммарная длина кода:
    ``M * log2(K)`` бит.

    Поиск через ADC (Asymmetric Distance Computation): для запроса считается
    M×K LUT расстояний до центроидов, далее расстояние до любого документа —
    это сумма M lookup'ов. Время поиска O(N × M) против O(N × D) у точного
    плотного поиска.
    """

    def __init__(
        self,
        input_dim: int,
        n_subspaces: int,
        n_centroids: int = 256,
        n_iter: int = 25,
        seed: int = 17,
    ) -> None:
        if input_dim % n_subspaces != 0:
            raise ValueError(
                f"input_dim={input_dim} не делится на n_subspaces={n_subspaces}"
            )
        if n_centroids > 2 ** 16:
            raise ValueError("n_centroids > 65536 не поддерживается этой реализацией")
        self.input_dim = int(input_dim)
        self.n_subspaces = int(n_subspaces)
        self.n_centroids = int(n_centroids)
        self.sub_dim = self.input_dim // self.n_subspaces
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.codebooks: list[np.ndarray] = []  # M массивов формы (K, sub_dim)
        self.last_fit_time_ms: float = 0.0

    @property
    def code_bits(self) -> int:
        return int(self.n_subspaces * np.log2(self.n_centroids))

    def _kmeans(self, X: np.ndarray, K: int, seed: int) -> np.ndarray:
        """Простая реализация K-means (Lloyd's algorithm)."""
        N, _ = X.shape
        rng = np.random.RandomState(seed)
        # Инициализация: случайно выбранные точки из X
        if N <= K:
            # Если данных меньше центроидов, повторяем
            init_idx = rng.choice(N, K, replace=True)
        else:
            init_idx = rng.choice(N, K, replace=False)
        centers = X[init_idx].astype(np.float32, copy=True)

        for _ in range(self.n_iter):
            # Векторизованное расстояние через (a-b)^2 = a^2 + b^2 - 2ab
            cn2 = (centers ** 2).sum(axis=1)             # (K,)
            xn2 = (X ** 2).sum(axis=1)                   # (N,)
            cross = X @ centers.T                        # (N, K)
            dists = xn2[:, None] + cn2[None, :] - 2 * cross
            labels = dists.argmin(axis=1)                # (N,)

            new_centers = np.zeros_like(centers)
            for k in range(K):
                mask = labels == k
                if mask.any():
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    new_centers[k] = centers[k]  # пустой кластер — оставляем как был
            centers = new_centers
        return centers

    def fit(self, embeddings: np.ndarray) -> "ProductQuantizer":
        started = time.perf_counter()
        X = embeddings.astype(np.float32, copy=False)
        self.codebooks = []
        for s in range(self.n_subspaces):
            sub = X[:, s * self.sub_dim : (s + 1) * self.sub_dim]
            book = self._kmeans(sub, self.n_centroids, self.seed + s)
            self.codebooks.append(book)
        self.last_fit_time_ms = (time.perf_counter() - started) * 1000.0
        return self

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """Возвращает коды формы (N, n_subspaces) с целыми значениями [0, K)."""
        X = embeddings.astype(np.float32, copy=False)
        dtype = np.uint8 if self.n_centroids <= 256 else np.uint16
        codes = np.zeros((len(X), self.n_subspaces), dtype=dtype)
        for s in range(self.n_subspaces):
            sub = X[:, s * self.sub_dim : (s + 1) * self.sub_dim]
            book = self.codebooks[s]
            cn2 = (book ** 2).sum(axis=1)
            sn2 = (sub ** 2).sum(axis=1)
            cross = sub @ book.T
            dists = sn2[:, None] + cn2[None, :] - 2 * cross
            codes[:, s] = dists.argmin(axis=1)
        return codes

    def search(
        self,
        query: np.ndarray,
        db_codes: np.ndarray,
        top_k: int,
    ) -> PQSearchResult:
        """ADC search: возвращает top_k документов с наименьшим приближённым L2."""
        # Строим LUT: distance(query_sub, centroid) для каждого подпространства
        lut = np.zeros((self.n_subspaces, self.n_centroids), dtype=np.float32)
        for s in range(self.n_subspaces):
            qsub = query[s * self.sub_dim : (s + 1) * self.sub_dim]
            book = self.codebooks[s]
            diff = book - qsub[None, :]
            lut[s] = (diff ** 2).sum(axis=-1)

        # Считаем расстояния до всех документов через LUT
        n = len(db_codes)
        dists = np.zeros(n, dtype=np.float32)
        for s in range(self.n_subspaces):
            dists += lut[s, db_codes[:, s]]

        if top_k >= n:
            order = np.argsort(dists, kind="stable")
        else:
            partition = np.argpartition(dists, top_k - 1)[:top_k]
            order = partition[np.argsort(dists[partition], kind="stable")]
        return PQSearchResult(indices=order, distances=dists[order])
