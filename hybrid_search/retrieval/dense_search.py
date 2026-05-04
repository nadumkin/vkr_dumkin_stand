from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from ..core.config import SearchConfig
from ..core.preprocessing import TextPreprocessor
from ..core.schemas import SearchResult, TextRecord
from ..core.utils import l2_normalize
from ..models.encoders import BaseTextEncoder


class DenseSearchPipeline:
    def __init__(
        self,
        preprocessor: TextPreprocessor,
        encoder: BaseTextEncoder,
    ) -> None:
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.records: list[TextRecord] = []
        self.dense_embeddings = np.empty((0, 0), dtype=np.float32)
        # Время кодирования корпуса (encoder forward + L2-нормализация).
        self.last_encode_time_ms: float = 0.0
        # Поэтапные тайминги последнего вызова search() (мс).
        self.last_search_timings: dict[str, float] = {
            "query_encode_ms": 0.0,
            "candidate_selection_ms": 0.0,
            "rerank_ms": 0.0,
            "total_ms": 0.0,
        }

    def __len__(self) -> int:
        return len(self.records)

    def index_documents(self, records: Sequence[TextRecord]) -> None:
        texts = [record.text for record in records]
        encode_started = time.perf_counter()
        dense_embeddings = self.encoder.encode(texts).astype(np.float32, copy=False)
        self.records = list(records)
        self.dense_embeddings = l2_normalize(dense_embeddings)
        self.last_encode_time_ms = (time.perf_counter() - encode_started) * 1000.0

    def memory_breakdown(self) -> dict:
        n = len(self.records)
        dense_bytes = int(self.dense_embeddings.nbytes)
        records_text_bytes = sum(
            len(record.text.encode("utf-8")) + len(str(record.metadata).encode("utf-8"))
            for record in self.records
        )
        total_bytes = dense_bytes + records_text_bytes
        return {
            "documents": n,
            "dense_embeddings_bytes": dense_bytes,
            "records_text_bytes": int(records_text_bytes),
            "total_bytes": int(total_bytes),
            "bytes_per_doc": int(total_bytes / n) if n > 0 else 0,
        }

    def search(
        self,
        query_text: str,
        config: SearchConfig | None = None,
    ) -> list[SearchResult]:
        timings = {"query_encode_ms": 0.0, "candidate_selection_ms": 0.0, "rerank_ms": 0.0, "total_ms": 0.0}
        total_started = time.perf_counter()
        try:
            if len(self.records) == 0:
                return []

            resolved = config or SearchConfig()
            top_k = max(int(resolved.top_k), 1)
            normalized_query = self.preprocessor.normalize(query_text)
            encode_started = time.perf_counter()
            dense_query = self.encoder.encode([normalized_query]).astype(np.float32, copy=False)
            dense_query = l2_normalize(dense_query)[0]
            timings["query_encode_ms"] = (time.perf_counter() - encode_started) * 1000.0

            scan_started = time.perf_counter()
            scores = self.dense_embeddings @ dense_query
            ranked = np.argsort(-scores, kind="stable")[:top_k]
            timings["candidate_selection_ms"] = (time.perf_counter() - scan_started) * 1000.0

            results: list[SearchResult] = []
            for rank, index in enumerate(ranked, start=1):
                record = self.records[int(index)]
                results.append(
                    SearchResult(
                        record_id=record.record_id,
                        text=record.text,
                        dense_score=float(scores[index]),
                        binary_distance=-1,
                        rank=rank,
                        metadata=dict(record.metadata),
                    )
                )
            return results
        finally:
            timings["total_ms"] = (time.perf_counter() - total_started) * 1000.0
            self.last_search_timings = timings
