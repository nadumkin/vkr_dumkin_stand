from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from ..core.config import EncoderConfig, HashingModelConfig, IndexConfig, SearchConfig, TrainingConfig
from ..core.preprocessing import TextPreprocessor
from ..io.data import load_jsonl, load_similarity_examples, load_text_records
from ..io.storage import load_json, save_json
from ..models.encoders import build_encoder
from ..models.hashing import HashingMLP, load_hash_checkpoint, save_hash_checkpoint
from ..models.training import HashingTrainer, build_triplet_embedding_dataset
from ..retrieval.evaluation import SearchEvaluator
from ..retrieval.indexing import BinaryCodeIndex
from ..retrieval.search import HybridSearchPipeline


def parse_hidden_dims(raw: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        if not raw:
            return ()
        return tuple(int(part) for part in raw.split(",") if part)
    return tuple(int(value) for value in raw)


def save_encoder_config(index_dir: str | Path, encoder_config: EncoderConfig) -> None:
    save_json(Path(index_dir) / "encoder_config.json", asdict(encoder_config))


def load_encoder_config(index_dir: str | Path) -> EncoderConfig:
    payload = load_json(Path(index_dir) / "encoder_config.json")
    return EncoderConfig(**payload)


def resolve_hash_model_config(model_config: HashingModelConfig, input_dim: int) -> HashingModelConfig:
    return replace(model_config, input_dim=int(input_dim), hidden_dims=tuple(model_config.hidden_dims))


@dataclass
class TrainingArtifacts:
    checkpoint: Path
    history: list[dict[str, Any]]
    encoder_config: EncoderConfig
    model_config: HashingModelConfig
    training_config: TrainingConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "history": self.history,
            "encoder_config": asdict(self.encoder_config),
            "model_config": self.model_config.to_dict(),
            "training_config": asdict(self.training_config),
        }


@dataclass
class IndexArtifacts:
    index_dir: Path
    documents: int
    index_backend_requested: str
    index_backend_active: str
    encoder_config: EncoderConfig
    index_config: IndexConfig
    encode_time_ms: float = 0.0
    hashing_time_ms: float = 0.0
    build_time_ms: float = 0.0
    memory: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_dir": str(self.index_dir),
            "documents": self.documents,
            "index_backend_requested": self.index_backend_requested,
            "index_backend_active": self.index_backend_active,
            "encoder_config": asdict(self.encoder_config),
            "index_config": asdict(self.index_config),
            "encode_time_ms": float(self.encode_time_ms),
            "hashing_time_ms": float(self.hashing_time_ms),
            "build_time_ms": float(self.build_time_ms),
            "memory": dict(self.memory),
        }


def train_hash_module(
    triplets_path: str | Path,
    checkpoint_path: str | Path,
    encoder_config: EncoderConfig,
    model_config: HashingModelConfig,
    training_config: TrainingConfig,
    preprocessor: TextPreprocessor | None = None,
) -> TrainingArtifacts:
    cleaner = preprocessor or TextPreprocessor()
    encoder = build_encoder(encoder_config, preprocessor=cleaner)
    examples = load_similarity_examples(triplets_path)
    dataset = build_triplet_embedding_dataset(examples, encoder, preprocessor=cleaner)
    resolved_model_config = resolve_hash_model_config(model_config, encoder.embedding_dim)
    model = HashingMLP(resolved_model_config)
    trainer = HashingTrainer(model, training_config)
    history = trainer.fit(dataset)

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_hash_checkpoint(
        checkpoint,
        model,
        extra={
            "encoder_config": asdict(encoder_config),
            "training_config": asdict(training_config),
            "training_history": history,
        },
    )
    return TrainingArtifacts(
        checkpoint=checkpoint,
        history=history,
        encoder_config=encoder_config,
        model_config=resolved_model_config,
        training_config=training_config,
    )


def build_index_from_corpus(
    corpus_path: str | Path,
    output_dir: str | Path,
    encoder_config: EncoderConfig,
    model_config: HashingModelConfig,
    index_config: IndexConfig,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    preprocessor: TextPreprocessor | None = None,
) -> IndexArtifacts:
    cleaner = preprocessor or TextPreprocessor()
    resolved_encoder_config = replace(encoder_config, device=device or encoder_config.device)
    encoder = build_encoder(resolved_encoder_config, preprocessor=cleaner)
    resolved_model_config = resolve_hash_model_config(model_config, encoder.embedding_dim)

    if checkpoint_path:
        model, _ = load_hash_checkpoint(checkpoint_path, device=resolved_encoder_config.device)
    else:
        model = HashingMLP(resolved_model_config)

    index = BinaryCodeIndex(
        code_bits=index_config.code_bits,
        backend=index_config.backend,
        hnsw_m=index_config.hnsw_m,
        hnsw_ef_construction=index_config.hnsw_ef_construction,
        hnsw_ef_search=index_config.hnsw_ef_search,
    )
    records = load_text_records(corpus_path)
    pipeline = HybridSearchPipeline(
        preprocessor=cleaner,
        encoder=encoder,
        hash_model=model,
        index=index,
        device=resolved_encoder_config.device,
    )
    pipeline.index_documents(records)

    index_dir = Path(output_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    pipeline.index.save(index_dir)
    save_hash_checkpoint(index_dir / "hash_model.pt", model, extra={"encoder_config": asdict(resolved_encoder_config)})
    save_encoder_config(index_dir, resolved_encoder_config)
    return IndexArtifacts(
        index_dir=index_dir,
        documents=len(records),
        index_backend_requested=index.requested_backend,
        index_backend_active=index.backend,
        encoder_config=resolved_encoder_config,
        index_config=index_config,
        encode_time_ms=pipeline.last_encode_time_ms,
        hashing_time_ms=pipeline.last_hashing_time_ms,
        build_time_ms=index.last_build_time_ms,
        memory=index.memory_breakdown(),
    )


def load_pipeline_from_index(
    index_dir: str | Path,
    device: str = "cpu",
    preprocessor: TextPreprocessor | None = None,
) -> HybridSearchPipeline:
    index_path = Path(index_dir)
    cleaner = preprocessor or TextPreprocessor()
    encoder_config = load_encoder_config(index_path)
    runtime_encoder_config = replace(encoder_config, device=device)
    encoder = build_encoder(runtime_encoder_config, preprocessor=cleaner)
    model, _ = load_hash_checkpoint(index_path / "hash_model.pt", device=device)
    index = BinaryCodeIndex.load(index_path)
    return HybridSearchPipeline(
        preprocessor=cleaner,
        encoder=encoder,
        hash_model=model,
        index=index,
        device=device,
    )


def evaluate_index(
    index_dir: str | Path,
    queries_path: str | Path,
    search_config: SearchConfig | None = None,
    device: str = "cpu",
    preprocessor: TextPreprocessor | None = None,
) -> dict[str, Any]:
    pipeline = load_pipeline_from_index(index_dir=index_dir, device=device, preprocessor=preprocessor)
    evaluator = SearchEvaluator(pipeline)
    return evaluator.evaluate(load_jsonl(queries_path), search_config=search_config)
