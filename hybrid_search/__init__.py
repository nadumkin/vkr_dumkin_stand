from importlib import import_module
import sys

from .core.config import (
    EncoderConfig,
    HashingModelConfig,
    IndexConfig,
    PreprocessingConfig,
    SearchConfig,
    TrainingConfig,
)
from .core.preprocessing import TextPreprocessor
from .core.schemas import SearchResult, SimilarityExample, TextRecord
from .experiments.dataset_preparation import DatasetPreparer
from .models.encoders import HashingTextEncoder, TransformerSentenceEncoder, build_encoder
from .models.hashing import HashingMLP, load_hash_checkpoint, save_hash_checkpoint
from .retrieval.indexing import BinaryCodeIndex
from .retrieval.search import HybridSearchPipeline

_LEGACY_MODULE_ALIASES = {
    "config": "hybrid_search.core.config",
    "preprocessing": "hybrid_search.core.preprocessing",
    "schemas": "hybrid_search.core.schemas",
    "utils": "hybrid_search.core.utils",
    "data": "hybrid_search.io.data",
    "storage": "hybrid_search.io.storage",
    "encoders": "hybrid_search.models.encoders",
    "hashing": "hybrid_search.models.hashing",
    "training": "hybrid_search.models.training",
    "indexing": "hybrid_search.retrieval.indexing",
    "search": "hybrid_search.retrieval.search",
    "evaluation": "hybrid_search.retrieval.evaluation",
    "dataset_preparation": "hybrid_search.experiments.dataset_preparation",
    "experiment_logging": "hybrid_search.experiments.experiment_logging",
    "experiment_runner": "hybrid_search.experiments.experiment_runner",
    "workflows": "hybrid_search.experiments.workflows",
}

for legacy_name, target in _LEGACY_MODULE_ALIASES.items():
    sys.modules.setdefault(f"{__name__}.{legacy_name}", import_module(target))

__all__ = [
    "BinaryCodeIndex",
    "DatasetPreparer",
    "EncoderConfig",
    "HashingMLP",
    "HashingModelConfig",
    "HashingTextEncoder",
    "HybridSearchPipeline",
    "IndexConfig",
    "PreprocessingConfig",
    "SearchConfig",
    "SearchResult",
    "SimilarityExample",
    "TextPreprocessor",
    "TextRecord",
    "TrainingConfig",
    "TransformerSentenceEncoder",
    "build_encoder",
    "load_hash_checkpoint",
    "save_hash_checkpoint",
]
