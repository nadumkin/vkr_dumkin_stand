from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from ..core.schemas import SimilarityExample, TextRecord


def load_jsonl(path: str | Path) -> List[dict]:
    payload: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload.append(json.loads(line))
    return payload


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_text_records(
    path: str | Path,
    id_field: str = "id",
    text_field: str = "text",
) -> List[TextRecord]:
    records: List[TextRecord] = []
    for row in load_jsonl(path):
        record_id = str(row[id_field])
        text = str(row[text_field])
        metadata = {key: value for key, value in row.items() if key not in {id_field, text_field}}
        records.append(TextRecord(record_id=record_id, text=text, metadata=metadata))
    return records


def load_similarity_examples(path: str | Path) -> List[SimilarityExample]:
    examples: List[SimilarityExample] = []
    for row in load_jsonl(path):
        examples.append(
            SimilarityExample(
                query_text=str(row["query"]),
                positive_text=str(row["positive"]),
                negative_text=row.get("negative"),
                query_id=row.get("query_id"),
                positive_id=row.get("positive_id"),
                negative_id=row.get("negative_id"),
                label=row.get("label"),
            )
        )
    return examples
