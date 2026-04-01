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


class SearchEvaluator:
    def __init__(self, pipeline: HybridSearchPipeline) -> None:
        self.pipeline = pipeline

    def evaluate(
        self,
        query_rows: Iterable[dict],
        search_config: SearchConfig | None = None,
    ) -> dict:
        rows = list(query_rows)
        if not rows:
            return {"queries": 0, "recall@k": 0.0, "map@k": 0.0, "ndcg@k": 0.0, "latency_ms": 0.0}

        resolved = search_config or SearchConfig()
        metrics: list[QueryEvaluation] = []
        for row in rows:
            query = str(row["query"])
            relevant = set(map(str, row["relevant_ids"]))
            started = time.perf_counter()
            results = self.pipeline.search(query, config=resolved)
            latency_ms = (time.perf_counter() - started) * 1000.0
            ranked_ids = [result.record_id for result in results]
            metrics.append(
                QueryEvaluation(
                    query=query,
                    recall=recall_at_k(ranked_ids, relevant, resolved.top_k),
                    map_k=average_precision_at_k(ranked_ids, relevant, resolved.top_k),
                    ndcg=ndcg_at_k(ranked_ids, relevant, resolved.top_k),
                    latency_ms=latency_ms,
                )
            )

        return {
            "queries": len(metrics),
            "recall@k": sum(metric.recall for metric in metrics) / len(metrics),
            "map@k": sum(metric.map_k for metric in metrics) / len(metrics),
            "ndcg@k": sum(metric.ndcg for metric in metrics) / len(metrics),
            "latency_ms": sum(metric.latency_ms for metric in metrics) / len(metrics),
        }
