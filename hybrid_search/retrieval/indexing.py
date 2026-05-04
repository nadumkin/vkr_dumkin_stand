from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..core.schemas import TextRecord
from ..core.utils import l2_normalize
from ..io.storage import ensure_directory, load_json, save_json
from ..models.hashing import pack_binary_codes

POPCOUNT_TABLE = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)


class BinaryCodeIndex:
    def __init__(
        self,
        code_bits: int,
        backend: str = "numpy",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
    ) -> None:
        self.code_bits = int(code_bits)
        self.requested_backend = backend
        self.backend = self._resolve_backend(backend)
        self.hnsw_m = int(hnsw_m)
        self.hnsw_ef_construction = int(hnsw_ef_construction)
        self.hnsw_ef_search = int(hnsw_ef_search)
        self.packed_codes = np.empty((0, (self.code_bits + 7) // 8), dtype=np.uint8)
        self.dense_embeddings = np.empty((0, 0), dtype=np.float32)
        self.records: list[TextRecord] = []
        self._hnsw_index = None
        # Время последней операции построения индекса (только сам индекс, без кодирования).
        self.last_build_time_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_backend(self, backend: str) -> str:
        if backend == "hnswlib":
            try:
                import hnswlib  # noqa: F401
            except ImportError:
                return "numpy"
        return backend

    def _binary_codes_to_bit_vectors(self, binary_codes: np.ndarray) -> np.ndarray:
        return (binary_codes > 0).astype(np.float32, copy=False)

    def _packed_codes_to_bit_vectors(self) -> np.ndarray:
        if len(self.packed_codes) == 0:
            return np.empty((0, self.code_bits), dtype=np.float32)
        unpacked = np.unpackbits(self.packed_codes, axis=1, bitorder="little")
        return unpacked[:, : self.code_bits].astype(np.float32, copy=False)

    def _rebuild_hnsw_index(self) -> None:
        self._hnsw_index = None
        if self.backend != "hnswlib" or len(self.records) == 0:
            return
        import hnswlib

        bit_vectors = self._packed_codes_to_bit_vectors()
        index = hnswlib.Index(space="l2", dim=self.code_bits)
        index.init_index(
            max_elements=len(bit_vectors),
            ef_construction=self.hnsw_ef_construction,
            M=self.hnsw_m,
        )
        index.add_items(bit_vectors, np.arange(len(bit_vectors), dtype=np.int32))
        index.set_ef(max(self.hnsw_ef_search, 1))
        self._hnsw_index = index

    def add(
        self,
        records: Sequence[TextRecord],
        binary_codes: np.ndarray,
        dense_embeddings: np.ndarray,
    ) -> None:
        if len(records) != len(binary_codes) or len(records) != len(dense_embeddings):
            raise ValueError("records, binary_codes and dense_embeddings must have the same length")
        packed = pack_binary_codes(binary_codes)
        normalized_dense = l2_normalize(dense_embeddings.astype(np.float32, copy=False))

        if len(self.records) == 0:
            self.packed_codes = packed
            self.dense_embeddings = normalized_dense
        else:
            self.packed_codes = np.vstack([self.packed_codes, packed])
            self.dense_embeddings = np.vstack([self.dense_embeddings, normalized_dense])
        self.records.extend(records)

        build_started = time.perf_counter()
        self._rebuild_hnsw_index()
        self.last_build_time_ms = (time.perf_counter() - build_started) * 1000.0

    def hamming_distances(self, query_code: np.ndarray) -> np.ndarray:
        packed_query = pack_binary_codes(query_code.reshape(1, -1))[0]
        xor = np.bitwise_xor(self.packed_codes, packed_query)
        return POPCOUNT_TABLE[xor].sum(axis=1).astype(np.int32)

    def cosine_scores(self, query_embedding: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
        query = l2_normalize(query_embedding.reshape(1, -1).astype(np.float32, copy=False))[0]
        pool = self.dense_embeddings if indices is None else self.dense_embeddings[indices]
        return pool @ query

    def candidate_indices(
        self,
        query_code: np.ndarray,
        top_n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return top-N candidate indices and their Hamming distances (aligned by position).

        The returned distances are aligned with ``candidate_indices`` (i.e. the i-th
        distance corresponds to the i-th candidate), not indexed by global document id.
        """
        if len(self.records) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int32)
        top_n = min(top_n, len(self.records))

        if self.backend == "hnswlib" and self._hnsw_index is not None:
            # HNSW with L2 space over {0,1} float vectors — squared L2 equals Hamming.
            query_vector = self._binary_codes_to_bit_vectors(query_code.reshape(1, -1))
            current_ef = max(self.hnsw_ef_search, top_n)
            self._hnsw_index.set_ef(current_ef)
            labels, hnsw_distances = self._hnsw_index.knn_query(query_vector, k=top_n)
            labels = labels[0]
            hnsw_distances = hnsw_distances[0]
            valid_mask = labels >= 0
            labels = labels[valid_mask]
            hnsw_distances = hnsw_distances[valid_mask]
            if len(labels) == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.int32)
            # hnswlib already returns labels sorted ascending by distance; squared L2
            # of {0,1} vectors equals Hamming distance, so we cast to int and trust it.
            return labels.astype(np.int64, copy=False), hnsw_distances.astype(np.int32, copy=False)

        distances = self.hamming_distances(query_code)
        partition = np.argpartition(distances, top_n - 1)[:top_n]
        ranking = partition[np.argsort(distances[partition], kind="stable")]
        return ranking.astype(np.int64), distances[ranking].astype(np.int32, copy=False)

    def memory_breakdown(self) -> dict:
        """Оценка занимаемой памяти основных структур индекса в байтах.

        Размер графа HNSW оценивается аналитически по формуле hnswlib
        ``per_element ≈ M_max0 * 2 * sizeof(uint32) + sizeof(label_t) + bit_vector``,
        что соответствует памяти на хранение нижнего уровня графа и метаданных.
        Это нижняя оценка: вышестоящие уровни занимают доли процента и в формулу
        не включены. Для backend ``numpy`` поле ``hnsw_graph_bytes`` равно нулю.
        """
        n = len(self.records)
        dense_bytes = int(self.dense_embeddings.nbytes)
        packed_bytes = int(self.packed_codes.nbytes)
        records_text_bytes = sum(
            len(record.text.encode("utf-8")) + len(str(record.metadata).encode("utf-8"))
            for record in self.records
        )
        if self.backend == "hnswlib" and n > 0:
            # M_max0 = 2*M (число рёбер на уровне 0 в hnswlib).
            edges_per_node = max(1, 2 * self.hnsw_m)
            graph_bytes = n * (edges_per_node * 4 + 4)  # uint32 per neighbour + label
            # Плюс хранение векторов внутри hnswlib (float32 на бит).
            vector_bytes = n * self.code_bits * 4
            hnsw_bytes = graph_bytes + vector_bytes
        else:
            hnsw_bytes = 0
        total_bytes = dense_bytes + packed_bytes + records_text_bytes + hnsw_bytes
        return {
            "documents": n,
            "code_bits": self.code_bits,
            "dense_embeddings_bytes": dense_bytes,
            "packed_codes_bytes": packed_bytes,
            "records_text_bytes": int(records_text_bytes),
            "hnsw_graph_bytes": hnsw_bytes,
            "total_bytes": int(total_bytes),
            "bytes_per_doc": int(total_bytes / n) if n > 0 else 0,
        }

    def save(self, directory: str | Path) -> None:
        target = ensure_directory(directory)
        np.savez_compressed(
            target / "index_arrays.npz",
            packed_codes=self.packed_codes,
            dense_embeddings=self.dense_embeddings,
        )
        payload = {
            "code_bits": self.code_bits,
            "backend": self.requested_backend,
            "active_backend": self.backend,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
            "records": [
                {
                    "id": record.record_id,
                    "text": record.text,
                    "metadata": record.metadata,
                }
                for record in self.records
            ],
        }
        save_json(target / "index_meta.json", payload)

    @classmethod
    def load(cls, directory: str | Path) -> "BinaryCodeIndex":
        target = Path(directory)
        meta = load_json(target / "index_meta.json")
        arrays = np.load(target / "index_arrays.npz")
        instance = cls(
            code_bits=int(meta["code_bits"]),
            backend=str(meta.get("backend", "numpy")),
            hnsw_m=int(meta.get("hnsw_m", 16)),
            hnsw_ef_construction=int(meta.get("hnsw_ef_construction", 200)),
            hnsw_ef_search=int(meta.get("hnsw_ef_search", 64)),
        )
        instance.packed_codes = arrays["packed_codes"].astype(np.uint8, copy=False)
        instance.dense_embeddings = arrays["dense_embeddings"].astype(np.float32, copy=False)
        instance.records = [
            TextRecord(record_id=row["id"], text=row["text"], metadata=row.get("metadata") or {})
            for row in meta["records"]
        ]
        instance._rebuild_hnsw_index()
        return instance
