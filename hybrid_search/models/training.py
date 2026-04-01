from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from typing import List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..core.config import TrainingConfig
from ..core.preprocessing import TextPreprocessor
from ..core.schemas import SimilarityExample
from .encoders import BaseTextEncoder
from .hashing import HashingMLP


class TripletEmbeddingDataset(Dataset):
    def __init__(self, queries: np.ndarray, positives: np.ndarray, negatives: np.ndarray) -> None:
        self.queries = torch.from_numpy(queries.astype(np.float32, copy=False))
        self.positives = torch.from_numpy(positives.astype(np.float32, copy=False))
        self.negatives = torch.from_numpy(negatives.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.queries.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.queries[index], self.positives[index], self.negatives[index]


def build_triplet_embedding_dataset(
    examples: Sequence[SimilarityExample],
    encoder: BaseTextEncoder,
    preprocessor: TextPreprocessor | None = None,
) -> TripletEmbeddingDataset:
    cleaner = preprocessor or TextPreprocessor()
    normalized_examples: List[tuple[str, str, str]] = []
    positive_pool: List[str] = []

    for example in examples:
        query = cleaner.normalize(example.query_text)
        positive = cleaner.normalize(example.positive_text)
        negative = cleaner.normalize(example.negative_text or "")
        normalized_examples.append((query, positive, negative))
        positive_pool.append(positive)

    materialized_examples: List[tuple[str, str, str]] = []
    for index, (query, positive, negative) in enumerate(normalized_examples):
        resolved_negative = negative
        if not resolved_negative:
            fallback_index = (index + 1) % max(len(positive_pool), 1)
            resolved_negative = positive_pool[fallback_index]
            if resolved_negative == positive and len(positive_pool) > 1:
                resolved_negative = positive_pool[(fallback_index + 1) % len(positive_pool)]
        materialized_examples.append((query, positive, resolved_negative))

    unique_texts = OrderedDict()
    for query, positive, negative in materialized_examples:
        unique_texts.setdefault(query, None)
        unique_texts.setdefault(positive, None)
        unique_texts.setdefault(negative, None)

    encoded = encoder.encode(list(unique_texts.keys()))
    lookup = {text: embedding for text, embedding in zip(unique_texts.keys(), encoded)}

    queries = np.stack([lookup[query] for query, _, _ in materialized_examples], axis=0)
    positives = np.stack([lookup[positive] for _, positive, _ in materialized_examples], axis=0)
    negatives = np.stack([lookup[negative] for _, _, negative in materialized_examples], axis=0)
    return TripletEmbeddingDataset(queries, positives, negatives)


class HashingTrainer:
    def __init__(self, model: HashingMLP, config: TrainingConfig | None = None) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

    def _hamming_proxy(self, left: torch.Tensor, right: torch.Tensor, code_bits: int) -> torch.Tensor:
        return 0.5 * (code_bits - (left * right).sum(dim=-1))

    def _triplet_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        code_bits = int(anchor.shape[-1])
        positive_distance = self._hamming_proxy(anchor, positive, code_bits)
        negative_distance = self._hamming_proxy(anchor, negative, code_bits)
        return torch.relu(positive_distance - negative_distance + self.config.margin).mean()

    def _quantization_loss(self, relaxed: torch.Tensor) -> torch.Tensor:
        return (1.0 - relaxed.abs()).pow(2).mean()

    def fit(self, dataset: TripletEmbeddingDataset) -> list[dict]:
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        history: list[dict] = []
        self.model.train()
        for epoch in range(1, self.config.epochs + 1):
            epoch_loss = 0.0
            epoch_triplet = 0.0
            epoch_quant = 0.0
            batches = 0
            for queries, positives, negatives in loader:
                queries = queries.to(self.device)
                positives = positives.to(self.device)
                negatives = negatives.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                query_out = self.model(queries, temperature=self.config.temperature)
                positive_out = self.model(positives, temperature=self.config.temperature)
                negative_out = self.model(negatives, temperature=self.config.temperature)

                triplet_loss = self._triplet_loss(
                    query_out.relaxed,
                    positive_out.relaxed,
                    negative_out.relaxed,
                )
                quantization_loss = (
                    self._quantization_loss(query_out.relaxed)
                    + self._quantization_loss(positive_out.relaxed)
                    + self._quantization_loss(negative_out.relaxed)
                ) / 3.0

                loss = triplet_loss + self.config.quantization_weight * quantization_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                optimizer.step()

                epoch_loss += float(loss.item())
                epoch_triplet += float(triplet_loss.item())
                epoch_quant += float(quantization_loss.item())
                batches += 1

            summary = {
                "epoch": epoch,
                "loss": epoch_loss / max(batches, 1),
                "triplet_loss": epoch_triplet / max(batches, 1),
                "quantization_loss": epoch_quant / max(batches, 1),
            }
            history.append(summary)
        return history
