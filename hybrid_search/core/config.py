from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class PreprocessingConfig:
    lowercase: bool = True
    strip_html: bool = True
    collapse_whitespace: bool = True
    normalize_unicode: bool = True
    max_length: int = 128


@dataclass
class EncoderConfig:
    backend: str = "hashing"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: str = "cpu"
    seed: int = 17


@dataclass
class HashingModelConfig:
    input_dim: int = 384
    code_bits: int = 128
    hidden_dims: Tuple[int, ...] = (256, 128)
    activation: str = "gelu"
    dropout: float = 0.1
    use_layer_norm: bool = True

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["hidden_dims"] = list(self.hidden_dims)
        return payload


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    margin: float = 4.0
    quantization_weight: float = 0.1
    gradient_clip_norm: float = 1.0
    temperature: float = 1.0
    device: str = "cpu"


@dataclass
class IndexConfig:
    code_bits: int = 128
    oversample_factor: int = 4
    binary_keep: int = 128
    backend: str = "numpy"
    dense_metric: str = "cosine"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64


@dataclass
class SearchConfig:
    strategy: str = "coarse_rerank"
    top_k: int = 10
    oversample_factor: int = 4
    max_hamming_distance: int | None = None


@dataclass
class PrototypeConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    hashing: HashingModelConfig = field(default_factory=HashingModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
