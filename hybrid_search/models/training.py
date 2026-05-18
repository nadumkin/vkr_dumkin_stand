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


class DistillationTrainer:
    """Hash-only обучение через self-distillation.

    Цель: бинарный код должен сохранять косинусное подобие dense-эмбеддингов.
    Для каждого батча конкатенируются векторы (q, p, n) → 3B сэмплов, считается
    матрица истинных косинусов dense_sim (3B × 3B, detached как teacher signal)
    и матрица предсказанных косинусов binary_sim в релаксированном бинарном
    пространстве. Лосс — MSE между ними на off-diagonal записях.

    Это плотный сигнал: каждый шаг даёт ~9B² пар обучения вместо одного
    triplet-неравенства. Энкодер не трогается — мы тренируем только хэш-голову,
    используя pretrained-эмбеддинги как фиксированный учитель.
    """

    def __init__(
        self,
        model: HashingMLP,
        config: TrainingConfig | None = None,
        distillation_weight: float = 1.0,
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.distillation_weight = float(distillation_weight)
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

    @staticmethod
    def _quantization_loss(relaxed: torch.Tensor) -> torch.Tensor:
        return (1.0 - relaxed.abs()).pow(2).mean()

    def fit(self, dataset: TripletEmbeddingDataset) -> list[dict]:
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Подготовка к temperature annealing: если ``temperature_end`` задан,
        # α(step) меняется линейно от ``temperature`` до ``temperature_end``.
        total_steps = max(1, self.config.epochs * len(loader))
        alpha_start = float(self.config.temperature)
        alpha_end = (
            float(self.config.temperature_end)
            if self.config.temperature_end is not None
            else alpha_start
        )

        history: list[dict] = []
        self.model.train()
        global_step = 0
        for epoch in range(1, self.config.epochs + 1):
            epoch_loss = 0.0
            epoch_distill = 0.0
            epoch_quant = 0.0
            batches = 0
            last_alpha = alpha_start
            for queries, positives, negatives in loader:
                global_step += 1
                # Линейное расписание α: на 1-м шаге = alpha_start, на последнем = alpha_end.
                progress = (global_step - 1) / max(total_steps - 1, 1)
                alpha = alpha_start + (alpha_end - alpha_start) * progress
                last_alpha = alpha

                # Все три ветви имеют один и тот же ground-truth «учитель» —
                # косинусное подобие исходных dense-эмбеддингов.
                h_all = torch.cat(
                    [
                        queries.to(self.device),
                        positives.to(self.device),
                        negatives.to(self.device),
                    ],
                    dim=0,
                )
                # Подстраховка: эмбеддинги из энкодера уже L2-нормированы, но
                # numerical drift могло их немного «увести».
                h_all = torch.nn.functional.normalize(h_all, p=2, dim=-1, eps=1e-6)

                optimizer.zero_grad(set_to_none=True)
                output = self.model(h_all, temperature=alpha)
                relaxed = output.relaxed  # (3B, code_bits) в [-1, +1]
                code_bits = int(relaxed.shape[-1])

                # Учитель: dense cosine, без градиентов
                with torch.no_grad():
                    target_sim = h_all @ h_all.T  # (3B, 3B), already L2-norm

                # Предсказание: бинарный cosine ≈ dot / B
                pred_sim = (relaxed @ relaxed.T) / float(code_bits)

                # Маска off-diagonal — диагональ всегда 1, не нужна для обучения
                size = h_all.shape[0]
                eye = torch.eye(size, device=self.device, dtype=torch.bool)

                distill_loss = ((pred_sim - target_sim) ** 2).masked_select(~eye).mean()
                quant_loss = self._quantization_loss(relaxed)

                loss = (
                    self.distillation_weight * distill_loss
                    + self.config.quantization_weight * quant_loss
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                optimizer.step()

                epoch_loss += float(loss.item())
                epoch_distill += float(distill_loss.item())
                epoch_quant += float(quant_loss.item())
                batches += 1

            summary = {
                "epoch": epoch,
                "loss": epoch_loss / max(batches, 1),
                "distillation_loss": epoch_distill / max(batches, 1),
                "quantization_loss": epoch_quant / max(batches, 1),
                "alpha_end_of_epoch": last_alpha,
            }
            history.append(summary)
        return history
