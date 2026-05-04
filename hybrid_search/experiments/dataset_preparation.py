from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core.preprocessing import TextPreprocessor
from ..io.data import write_jsonl
from ..io.storage import ensure_directory, save_json


def _take_limit(rows: Sequence[dict], limit: int | None) -> list[dict]:
    materialized = list(rows)
    return materialized if limit is None else materialized[:limit]


class DatasetPreparer:
    def __init__(
        self,
        preprocessor: TextPreprocessor | None = None,
        stsb_positive_threshold: float = 4.0,
        stsb_negative_threshold: float = 2.0,
        holdout_fraction: float = 0.5,
        seed: int = 17,
    ) -> None:
        if not 0.0 < float(holdout_fraction) < 1.0:
            raise ValueError("holdout_fraction must be between 0 and 1.")
        self.preprocessor = preprocessor or TextPreprocessor()
        self.stsb_positive_threshold = float(stsb_positive_threshold)
        self.stsb_negative_threshold = float(stsb_negative_threshold)
        self.holdout_fraction = float(holdout_fraction)
        self.seed = int(seed)
        self.triplet_rng = random.Random(seed)
        self.split_rng = random.Random(seed + 1)

    def prepare_from_hf(
        self,
        output_dir: str | Path,
        dataset_names: Sequence[str] = ("allnli", "stsb", "qqp"),
        cache_dir: str | Path | None = None,
        limit_train_rows: int | None = None,
        limit_eval_rows: int | None = None,
    ) -> dict:
        datasets_by_name = self.load_supported_hf_datasets(
            dataset_names=dataset_names,
            cache_dir=cache_dir,
            limit_train_rows=limit_train_rows,
            limit_eval_rows=limit_eval_rows,
        )
        return self.prepare_from_splits(output_dir=output_dir, datasets_by_name=datasets_by_name)

    def prepare_from_splits(
        self,
        output_dir: str | Path,
        datasets_by_name: Mapping[str, Mapping[str, Sequence[dict]]],
    ) -> dict:
        self.triplet_rng = random.Random(self.seed)
        self.split_rng = random.Random(self.seed + 1)
        output_path = ensure_directory(output_dir)
        corpus_texts: dict[str, dict[str, Any]] = {}
        train_triplets: list[dict[str, Any]] = []
        evaluation_queries: list[dict[str, Any]] = []
        dataset_stats: dict[str, dict[str, int]] = {}

        for dataset_name, splits in datasets_by_name.items():
            dataset_key = dataset_name.lower().strip()
            dataset_stats[dataset_key] = {split_name: len(list(rows)) for split_name, rows in splits.items()}
            if dataset_key == "allnli":
                for split_name, rows in splits.items():
                    rows_list = list(rows)
                    self._collect_allnli_corpus_rows(corpus_texts, rows_list, dataset_key, split_name)
                    if split_name == "train":
                        train_triplets.extend(self._build_allnli_triplets(rows_list, dataset_key))
            elif dataset_key == "stsb":
                for split_name, rows in splits.items():
                    rows_list = list(rows)
                    self._collect_pair_corpus_rows(corpus_texts, rows_list, dataset_key, split_name, "sentence1", "sentence2")
                    if split_name == "train":
                        train_triplets.extend(self._build_stsb_triplets(rows_list, dataset_key))
                    elif split_name in {"validation", "test"}:
                        evaluation_queries.extend(
                            self._build_pair_query_rows(
                                rows_list,
                                corpus_texts,
                                dataset_key,
                                "sentence1",
                                "sentence2",
                                "label",
                                self.stsb_positive_threshold,
                            )
                        )
            elif dataset_key == "qqp":
                for split_name, rows in splits.items():
                    rows_list = list(rows)
                    self._collect_pair_corpus_rows(corpus_texts, rows_list, dataset_key, split_name, "question1", "question2")
                    if split_name == "train":
                        train_triplets.extend(self._build_binary_pair_triplets(rows_list, dataset_key, "question1", "question2"))
                    elif split_name in {"validation", "test"}:
                        evaluation_queries.extend(
                            self._build_pair_query_rows(
                                rows_list,
                                corpus_texts,
                                dataset_key,
                                "question1",
                                "question2",
                                "label",
                                1.0,
                            )
                        )
            else:
                raise ValueError(f"Unsupported dataset name: {dataset_name}")

        corpus_rows = sorted(corpus_texts.values(), key=lambda row: row["id"])
        train_triplets = self._deduplicate_rows(train_triplets, ("query", "positive", "negative"))
        merged_eval_queries = self._merge_query_rows(evaluation_queries)
        val_queries, test_queries = self._split_query_rows(merged_eval_queries)

        write_jsonl(output_path / "corpus.jsonl", corpus_rows)
        write_jsonl(output_path / "train_triplets.jsonl", train_triplets)
        write_jsonl(output_path / "val_queries.jsonl", val_queries)
        write_jsonl(output_path / "test_queries.jsonl", test_queries)

        summary = {
            "datasets": dataset_stats,
            "corpus_size": len(corpus_rows),
            "train_triplets": len(train_triplets),
            "evaluation_pool_queries": len(merged_eval_queries),
            "val_queries": len(val_queries),
            "test_queries": len(test_queries),
            "split_strategy": {
                "type": "holdout_eval_pool",
                "holdout_fraction": self.holdout_fraction,
                "seed": self.seed + 1,
            },
            "files": {
                "corpus": str(output_path / "corpus.jsonl"),
                "train_triplets": str(output_path / "train_triplets.jsonl"),
                "val_queries": str(output_path / "val_queries.jsonl"),
                "test_queries": str(output_path / "test_queries.jsonl"),
            },
        }
        save_json(output_path / "dataset_summary.json", summary)
        return summary

    def load_supported_hf_datasets(
        self,
        dataset_names: Sequence[str],
        cache_dir: str | Path | None = None,
        limit_train_rows: int | None = None,
        limit_eval_rows: int | None = None,
    ) -> dict[str, dict[str, list[dict]]]:
        try:
            from datasets import concatenate_datasets, load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is not installed. Install project dependencies in .venv before running prepare-datasets."
            ) from exc

        loaded: dict[str, dict[str, list[dict]]] = {}
        cache_arg = str(cache_dir) if cache_dir else None
        for dataset_name in dataset_names:
            dataset_key = dataset_name.lower().strip()
            if dataset_key == "allnli":
                snli = load_dataset("snli", cache_dir=cache_arg)
                mnli = load_dataset("multi_nli", cache_dir=cache_arg)
                loaded[dataset_key] = {
                    "train": _take_limit(
                        self._concatenate_available_splits(
                            concatenate_datasets,
                            (snli, mnli),
                            ("train",),
                        ),
                        limit_train_rows,
                    ),
                    "validation": _take_limit(
                        self._concatenate_available_splits(
                            concatenate_datasets,
                            (snli, mnli),
                            ("validation", "validation_matched", "validation_mismatched"),
                        ),
                        limit_eval_rows,
                    ),
                    "test": _take_limit(
                        self._concatenate_available_splits(
                            concatenate_datasets,
                            (snli, mnli),
                            ("test", "test_matched", "test_mismatched"),
                        ),
                        limit_eval_rows,
                    ),
                }
            elif dataset_key == "stsb":
                dataset = load_dataset("glue", "stsb", cache_dir=cache_arg)
                loaded[dataset_key] = {
                    "train": _take_limit(dataset["train"], limit_train_rows),
                    "validation": _take_limit(dataset["validation"], limit_eval_rows),
                    "test": _take_limit(dataset["test"], limit_eval_rows),
                }
            elif dataset_key == "qqp":
                dataset = load_dataset("glue", "qqp", cache_dir=cache_arg)
                loaded[dataset_key] = {
                    "train": _take_limit(dataset["train"], limit_train_rows),
                    "validation": _take_limit(dataset["validation"], limit_eval_rows),
                    "test": _take_limit(dataset["test"], limit_eval_rows),
                }
            else:
                raise ValueError(f"Unsupported dataset name: {dataset_name}")
        return loaded

    def _concatenate_available_splits(
        self,
        concatenate_datasets: Any,
        dataset_dicts: Sequence[Mapping[str, Sequence[dict]]],
        split_names: Sequence[str],
    ) -> Sequence[dict]:
        available_parts: list[Sequence[dict]] = []
        for dataset_dict in dataset_dicts:
            for split_name in split_names:
                if split_name in dataset_dict:
                    available_parts.append(dataset_dict[split_name])
        if not available_parts:
            return []
        if len(available_parts) == 1:
            return available_parts[0]
        return concatenate_datasets(list(available_parts))

    def _normalize_text(self, text: Any) -> str:
        return self.preprocessor.normalize("" if text is None else str(text))

    def _make_text_id(self, text: str) -> str:
        return f"txt_{hashlib.blake2b(text.encode('utf-8'), digest_size=8).hexdigest()}"

    def _register_corpus_text(
        self,
        corpus_texts: dict[str, dict[str, Any]],
        text: Any,
        dataset_name: str,
        split_name: str,
    ) -> str | None:
        normalized = self._normalize_text(text)
        if not normalized:
            return None
        record = corpus_texts.setdefault(
            normalized,
            {
                "id": self._make_text_id(normalized),
                "text": normalized,
                "metadata": {"datasets": [], "splits": []},
            },
        )
        if dataset_name not in record["metadata"]["datasets"]:
            record["metadata"]["datasets"].append(dataset_name)
        if split_name not in record["metadata"]["splits"]:
            record["metadata"]["splits"].append(split_name)
        return record["id"]

    def _extract_first(self, row: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in row and row[key] is not None:
                return row[key]
        return None

    def _label_to_name(self, row: Mapping[str, Any]) -> str | None:
        raw = row.get("label")
        if raw is None:
            return None
        if isinstance(raw, str):
            return raw.lower()
        mapping = {0: "entailment", 1: "neutral", 2: "contradiction", -1: "unknown"}
        return mapping.get(int(raw))

    def _choose_negative(self, local_negatives: list[str], global_negatives: list[str], positive_text: str) -> str | None:
        pool = [text for text in local_negatives if text != positive_text]
        if not pool:
            pool = [text for text in global_negatives if text != positive_text]
        if not pool:
            return None
        return self.triplet_rng.choice(pool)

    def _collect_allnli_corpus_rows(
        self,
        corpus_texts: dict[str, dict[str, Any]],
        rows: Sequence[dict],
        dataset_name: str,
        split_name: str,
    ) -> None:
        for row in rows:
            self._register_corpus_text(corpus_texts, self._extract_first(row, "premise", "sentence1"), dataset_name, split_name)
            self._register_corpus_text(corpus_texts, self._extract_first(row, "hypothesis", "sentence2"), dataset_name, split_name)

    def _collect_pair_corpus_rows(
        self,
        corpus_texts: dict[str, dict[str, Any]],
        rows: Sequence[dict],
        dataset_name: str,
        split_name: str,
        left_key: str,
        right_key: str,
    ) -> None:
        for row in rows:
            self._register_corpus_text(corpus_texts, row.get(left_key), dataset_name, split_name)
            self._register_corpus_text(corpus_texts, row.get(right_key), dataset_name, split_name)

    def _build_allnli_triplets(self, rows: Sequence[dict], dataset_name: str) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"positives": [], "negatives": []})
        global_negatives: list[str] = []
        for row in rows:
            label_name = self._label_to_name(row)
            premise = self._normalize_text(self._extract_first(row, "premise", "sentence1"))
            hypothesis = self._normalize_text(self._extract_first(row, "hypothesis", "sentence2"))
            if not premise or not hypothesis or label_name == "unknown":
                continue
            if label_name == "entailment":
                grouped[premise]["positives"].append(hypothesis)
            elif label_name in {"neutral", "contradiction"}:
                grouped[premise]["negatives"].append(hypothesis)
                global_negatives.append(hypothesis)

        triplets: list[dict[str, Any]] = []
        for query, bucket in grouped.items():
            for positive in bucket["positives"]:
                negative = self._choose_negative(bucket["negatives"], global_negatives, positive)
                if negative is None:
                    continue
                triplets.append({"query": query, "positive": positive, "negative": negative, "source_dataset": dataset_name})
        return triplets

    def _build_binary_pair_triplets(
        self,
        rows: Sequence[dict],
        dataset_name: str,
        left_key: str,
        right_key: str,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"positives": [], "negatives": []})
        global_negatives: list[str] = []
        for row in rows:
            query = self._normalize_text(row.get(left_key))
            candidate = self._normalize_text(row.get(right_key))
            label = row.get("label")
            if not query or not candidate or label is None:
                continue
            if int(label) == 1:
                grouped[query]["positives"].append(candidate)
            else:
                grouped[query]["negatives"].append(candidate)
                global_negatives.append(candidate)

        triplets: list[dict[str, Any]] = []
        for query, bucket in grouped.items():
            for positive in bucket["positives"]:
                negative = self._choose_negative(bucket["negatives"], global_negatives, positive)
                if negative is None:
                    continue
                triplets.append({"query": query, "positive": positive, "negative": negative, "source_dataset": dataset_name})
        return triplets

    def _build_stsb_triplets(self, rows: Sequence[dict], dataset_name: str) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"positives": [], "negatives": []})
        global_negatives: list[str] = []
        for row in rows:
            query = self._normalize_text(row.get("sentence1"))
            candidate = self._normalize_text(row.get("sentence2"))
            score = row.get("label")
            if not query or not candidate or score is None:
                continue
            score_value = float(score)
            if score_value >= self.stsb_positive_threshold:
                grouped[query]["positives"].append(candidate)
            elif score_value <= self.stsb_negative_threshold:
                grouped[query]["negatives"].append(candidate)
                global_negatives.append(candidate)

        triplets: list[dict[str, Any]] = []
        for query, bucket in grouped.items():
            for positive in bucket["positives"]:
                negative = self._choose_negative(bucket["negatives"], global_negatives, positive)
                if negative is None:
                    continue
                triplets.append(
                    {
                        "query": query,
                        "positive": positive,
                        "negative": negative,
                        "label": 1.0,
                        "source_dataset": dataset_name,
                    }
                )
        return triplets

    def _build_pair_query_rows(
        self,
        rows: Sequence[dict],
        corpus_texts: Mapping[str, dict[str, Any]],
        dataset_name: str,
        left_key: str,
        right_key: str,
        label_key: str,
        positive_threshold: float,
    ) -> list[dict[str, Any]]:
        query_to_relevant: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            query = self._normalize_text(row.get(left_key))
            candidate = self._normalize_text(row.get(right_key))
            label = row.get(label_key)
            if not query or not candidate or label is None:
                continue
            if float(label) < positive_threshold:
                continue
            record = corpus_texts.get(candidate)
            if record is None:
                continue
            query_to_relevant[query].add(record["id"])

        return [
            {
                "query": query,
                "relevant_ids": sorted(relevant_ids),
                "source_dataset": dataset_name,
            }
            for query, relevant_ids in query_to_relevant.items()
            if relevant_ids
        ]

    def _deduplicate_rows(self, rows: Sequence[dict[str, Any]], key_fields: Sequence[str]) -> list[dict[str, Any]]:
        seen = set()
        output = []
        for row in rows:
            key = tuple(row.get(field) for field in key_fields)
            if key in seen:
                continue
            seen.add(key)
            output.append(row)
        return output

    def _merge_query_rows(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for row in rows:
            entry = merged.setdefault(
                row["query"],
                {"query": row["query"], "relevant_ids": set(), "source_datasets": set()},
            )
            entry["relevant_ids"].update(row.get("relevant_ids", []))
            if row.get("source_dataset"):
                entry["source_datasets"].add(row["source_dataset"])

        result = []
        for entry in merged.values():
            result.append(
                {
                    "query": entry["query"],
                    "relevant_ids": sorted(entry["relevant_ids"]),
                    "source_datasets": sorted(entry["source_datasets"]),
                }
            )
        result.sort(key=lambda row: row["query"])
        return result

    def _split_query_rows(self, rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        materialized = list(rows)
        if len(materialized) <= 1:
            return materialized, []

        shuffled = list(materialized)
        self.split_rng.shuffle(shuffled)
        test_size = int(round(len(shuffled) * self.holdout_fraction))
        test_size = min(max(test_size, 1), len(shuffled) - 1)
        test_rows = sorted(shuffled[:test_size], key=lambda row: row["query"])
        val_rows = sorted(shuffled[test_size:], key=lambda row: row["query"])
        return val_rows, test_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare AllNLI/STS-B/QQP datasets.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default="allnli,stsb,qqp")
    parser.add_argument("--cache-dir")
    parser.add_argument("--limit-train-rows", type=int)
    parser.add_argument("--limit-eval-rows", type=int)
    parser.add_argument("--stsb-positive-threshold", type=float, default=4.0)
    parser.add_argument("--stsb-negative-threshold", type=float, default=2.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    preparer = DatasetPreparer(
        stsb_positive_threshold=args.stsb_positive_threshold,
        stsb_negative_threshold=args.stsb_negative_threshold,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    summary = preparer.prepare_from_hf(
        output_dir=args.output_dir,
        dataset_names=[item.strip() for item in args.datasets.split(",") if item.strip()],
        cache_dir=args.cache_dir,
        limit_train_rows=args.limit_train_rows,
        limit_eval_rows=args.limit_eval_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
