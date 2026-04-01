from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TextRecord:
    record_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimilarityExample:
    query_text: str
    positive_text: str
    negative_text: Optional[str] = None
    query_id: Optional[str] = None
    positive_id: Optional[str] = None
    negative_id: Optional[str] = None
    label: Optional[float] = None


@dataclass
class SearchResult:
    record_id: str
    text: str
    dense_score: float
    binary_distance: int
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncodedCollection:
    ids: List[str]
    texts: List[str]
    dense_embeddings: np.ndarray
    binary_codes: np.ndarray
    metadata: List[Dict[str, Any]] = field(default_factory=list)
