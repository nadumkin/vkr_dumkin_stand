from __future__ import annotations

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
        self._rebuild_hnsw_index()

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
        distances = self.hamming_distances(query_code)
        if len(distances) == 0:
            return np.array([], dtype=np.int64), distances
        top_n = min(top_n, len(distances))
        if self.backend == "hnswlib" and self._hnsw_index is not None:
            query_vector = self._binary_codes_to_bit_vectors(query_code.reshape(1, -1))
            labels, _ = self._hnsw_index.knn_query(query_vector, k=top_n)
            ranking = labels[0]
            ranking = ranking[ranking >= 0]
            if len(ranking) > 0:
                ranking = ranking[np.argsort(distances[ranking], kind="stable")]
                return ranking.astype(np.int64), distances
        partition = np.argpartition(distances, top_n - 1)[:top_n]
        ranking = partition[np.argsort(distances[partition], kind="stable")]
        return ranking.astype(np.int64), distances

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
