from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

from ..core.config import SearchConfig
from .search import HybridSearchPipeline


def recall_at_k(results: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(results[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def average_precision_at_k(results: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, record_id in enumerate(results[:k], start=1):
        if record_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant_ids)


def ndcg_at_k(results: Sequence[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for rank, record_id in enumerate(results[:k], start=1):
        relevance = 1.0 if record_id in relevant_ids else 0.0
        if relevance:
            dcg += relevance / math.log2(rank + 1)
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg


@dataclass
class QueryEvaluation:
    query: str
    recall: float
    map_k: float
    ndcg: float
    latency_ms: float
    query_encode_ms: float
    candidate_selection_ms: float
    rerank_ms: float


class SearchEvaluator:
    def __init__(self, pipeline) -> None:  # HybridSearchPipeline | DenseSearchPipeline
        self.pipeline = pipeline

    def evaluate(
        self,
        query_rows: Iterable[dict],
        search_config: SearchConfig | None = None,
        warmup_queries: int = 0,
    ) -> dict:
        rows = list(query_rows)
        if not rows:
            return {
                "queries": 0,
                "recall@k": 0.0,
                "map@k": 0.0,
                "ndcg@k": 0.0,
                "latency_ms": 0.0,
                "latency_breakdown_ms": {
                    "query_encode_ms": 0.0,
                    "candidate_selection_ms": 0.0,
                    "rerank_ms": 0.0,
                },
            }

        resolved = search_config or SearchConfig()

        # Прогрев: первые ``warmup_queries`` запросов исполняются, но не учитываются
        # в итоговой статистике задержек, чтобы исключить накладные расходы инициализации.
        for row in rows[: max(0, warmup_queries)]:
            self.pipeline.search(str(row["query"]), config=resolved)

        metrics: list[QueryEvaluation] = []
        for row in rows:
            query = str(row["query"])
            relevant = set(map(str, row["relevant_ids"]))
            started = time.perf_counter()
            results = self.pipeline.search(query, config=resolved)
            latency_ms = (time.perf_counter() - started) * 1000.0
            stage = getattr(self.pipeline, "last_search_timings", {}) or {}
            ranked_ids = [result.record_id for result in results]
            metrics.append(
                QueryEvaluation(
                    query=query,
                    recall=recall_at_k(ranked_ids, relevant, resolved.top_k),
                    map_k=average_precision_at_k(ranked_ids, relevant, resolved.top_k),
                    ndcg=ndcg_at_k(ranked_ids, relevant, resolved.top_k),
                    latency_ms=latency_ms,
                    query_encode_ms=float(stage.get("query_encode_ms", 0.0)),
                    candidate_selection_ms=float(stage.get("candidate_selection_ms", 0.0)),
                    rerank_ms=float(stage.get("rerank_ms", 0.0)),
                )
            )

        n = len(metrics)
        return {
            "queries": n,
            "recall@k": sum(m.recall for m in metrics) / n,
            "map@k": sum(m.map_k for m in metrics) / n,
            "ndcg@k": sum(m.ndcg for m in metrics) / n,
            "latency_ms": sum(m.latency_ms for m in metrics) / n,
            "latency_breakdown_ms": {
                "query_encode_ms": sum(m.query_encode_ms for m in metrics) / n,
                "candidate_selection_ms": sum(m.candidate_selection_ms for m in metrics) / n,
                "rerank_ms": sum(m.rerank_ms for m in metrics) / n,
            },
        }
