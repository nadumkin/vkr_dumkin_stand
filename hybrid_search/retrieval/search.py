from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch

from ..core.config import SearchConfig
from ..core.preprocessing import TextPreprocessor
from ..core.schemas import SearchResult, TextRecord
from ..models.encoders import BaseTextEncoder
from ..models.hashing import HashingMLP
from .indexing import BinaryCodeIndex


class HybridSearchPipeline:
    def __init__(
        self,
        preprocessor: TextPreprocessor,
        encoder: BaseTextEncoder,
        hash_model: HashingMLP,
        index: BinaryCodeIndex | None = None,
        device: str = "cpu",
    ) -> None:
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.hash_model = hash_model.to(device)
        self.index = index if index is not None else BinaryCodeIndex(code_bits=hash_model.config.code_bits)
        self.device = device
        # Тайминги последней операции index_documents (мс).
        self.last_encode_time_ms: float = 0.0
        self.last_hashing_time_ms: float = 0.0
        # Поэтапные тайминги последнего вызова search() (мс).
        self.last_search_timings: dict[str, float] = {
            "query_encode_ms": 0.0,
            "candidate_selection_ms": 0.0,
            "rerank_ms": 0.0,
            "total_ms": 0.0,
        }

    def index_documents(self, records: Sequence[TextRecord]) -> None:
        texts = [record.text for record in records]
        encode_started = time.perf_counter()
        dense_embeddings = self.encoder.encode(texts)
        self.last_encode_time_ms = (time.perf_counter() - encode_started) * 1000.0
        hashing_started = time.perf_counter()
        binary_codes = self.hash_model.encode_embeddings(dense_embeddings, device=self.device)
        self.last_hashing_time_ms = (time.perf_counter() - hashing_started) * 1000.0
        self.index.add(records=records, binary_codes=binary_codes, dense_embeddings=dense_embeddings)

    def encode_query(self, query_text: str) -> tuple[np.ndarray, np.ndarray]:
        dense_embedding = self.encoder.encode([query_text])[0]
        binary_code = self.hash_model.encode_embeddings(dense_embedding.reshape(1, -1), device=self.device)[0]
        return dense_embedding.astype(np.float32, copy=False), binary_code.astype(np.int8, copy=False)

    def _build_results(
        self,
        candidate_indices: np.ndarray,
        dense_scores: np.ndarray,
        binary_distances: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        """Rank candidates by dense score and assemble SearchResult objects.

        ``binary_distances`` must be aligned positionally with ``candidate_indices``
        (i.e. the i-th value is the Hamming distance of the i-th candidate to the query).
        """
        ranked_local = np.argsort(-dense_scores, kind="stable")[:top_k]
        results: list[SearchResult] = []
        for rank, local_index in enumerate(ranked_local, start=1):
            global_index = int(candidate_indices[local_index])
            record = self.index.records[global_index]
            results.append(
                SearchResult(
                    record_id=record.record_id,
                    text=record.text,
                    dense_score=float(dense_scores[local_index]),
                    binary_distance=int(binary_distances[local_index]),
                    rank=rank,
                    metadata=dict(record.metadata),
                )
            )
        return results

    def search(
        self,
        query_text: str,
        config: SearchConfig | None = None,
    ) -> list[SearchResult]:
        timings = {"query_encode_ms": 0.0, "candidate_selection_ms": 0.0, "rerank_ms": 0.0, "total_ms": 0.0}
        total_started = time.perf_counter()
        try:
            if len(self.index) == 0:
                return []
            resolved = config or SearchConfig()

            encode_started = time.perf_counter()
            dense_query, binary_query = self.encode_query(self.preprocessor.normalize(query_text))
            timings["query_encode_ms"] = (time.perf_counter() - encode_started) * 1000.0

            top_k = max(int(resolved.top_k), 1)
            oversample = max(int(resolved.oversample_factor), 1)

            if resolved.strategy == "coarse_rerank":
                cand_started = time.perf_counter()
                candidate_indices, candidate_distances = self.index.candidate_indices(
                    binary_query, top_k * oversample
                )
                timings["candidate_selection_ms"] = (time.perf_counter() - cand_started) * 1000.0
                if len(candidate_indices) == 0:
                    return []
                rerank_started = time.perf_counter()
                dense_scores = self.index.cosine_scores(dense_query, indices=candidate_indices)
                results = self._build_results(candidate_indices, dense_scores, candidate_distances, top_k)
                timings["rerank_ms"] = (time.perf_counter() - rerank_started) * 1000.0
                return results

            if resolved.strategy == "integrated_filtering":
                cand_started = time.perf_counter()
                full_distances = self.index.hamming_distances(binary_query)
                if resolved.max_hamming_distance is not None:
                    candidate_indices = np.flatnonzero(full_distances <= resolved.max_hamming_distance)
                else:
                    shortlist = max(top_k * oversample, top_k)
                    top_by_hamming, _ = self.index.candidate_indices(binary_query, shortlist)
                    if len(top_by_hamming) == 0:
                        timings["candidate_selection_ms"] = (time.perf_counter() - cand_started) * 1000.0
                        return []
                    threshold = int(full_distances[top_by_hamming[-1]])
                    candidate_indices = np.flatnonzero(full_distances <= threshold)
                timings["candidate_selection_ms"] = (time.perf_counter() - cand_started) * 1000.0
                if len(candidate_indices) == 0:
                    return []
                rerank_started = time.perf_counter()
                candidate_distances = full_distances[candidate_indices].astype(np.int32, copy=False)
                dense_scores = self.index.cosine_scores(dense_query, indices=candidate_indices)
                results = self._build_results(candidate_indices, dense_scores, candidate_distances, top_k)
                timings["rerank_ms"] = (time.perf_counter() - rerank_started) * 1000.0
                return results

            raise ValueError(f"Unsupported search strategy: {resolved.strategy}")
        finally:
            timings["total_ms"] = (time.perf_counter() - total_started) * 1000.0
            self.last_search_timings = timings
