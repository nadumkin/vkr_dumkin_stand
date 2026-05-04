"""Совместное дообучение трансформер-энкодера и хэширующего модуля.

В отличие от ``models.training.HashingTrainer`` (которая работает на
зафиксированных эмбеддингах), ``JointTrainer`` пропускает градиенты
через весь конвейер: текст → токены → BERT → mean-pool → MLP → relaxed
бинарный код → triplet-loss + штраф квантизации.

Особенности промышленной реализации:

- **Дискриминативные learning rate** — отдельные группы параметров для
  энкодера (обычно 1e-5) и для хэш-головы (1e-3); это стандартная практика
  при fine-tune предобученных трансформеров.
- **Linear warmup + cosine decay** для устойчивого старта на ранних шагах.
- **AMP / mixed precision** под CUDA (на MPS пока отключаем — не все ядра
  стабильно работают в bfloat16).
- **Pre-tokenization** триплетов один раз на старте: всё помещается в RAM
  (~55 МБ на 36K триплетов × 3 текста × 128 токенов × 4 байта).
- **Чекпойнты** содержат состояние энкодера, хэш-головы, оптимизатора и
  шедулера — обучение можно возобновить.
- **Eval-хук** между эпохами: вызывает заданный callback, чтобы строить
  индекс на текущем состоянии модели и считать recall@k на val_queries.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..core.schemas import SimilarityExample
from .encoders import TransformerSentenceEncoder
from .hashing import HashingMLP


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
@dataclass
class JointTrainingConfig:
    epochs: int = 3
    batch_size: int = 64
    encoder_lr: float = 2e-5
    hash_lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    margin: float = 4.0
    quantization_weight: float = 0.1
    temperature: float = 1.0
    gradient_clip_norm: float = 1.0
    max_seq_length: int = 128
    mixed_precision: bool = False
    seed: int = 17
    log_every_n_steps: int = 50
    eval_every_n_epochs: int = 1
    save_every_n_epochs: int = 1
    num_workers: int = 0
    pin_memory: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Датасет
# ---------------------------------------------------------------------------
class TextTripletDataset(Dataset):
    """Хранит пред-токенизированные триплеты как плотные тензоры.

    На входе ожидает список ``SimilarityExample``; токенизация выполняется
    в конструкторе один раз. Это в разы быстрее токенизации в DataLoader.
    """

    def __init__(
        self,
        examples: Sequence[SimilarityExample],
        encoder: TransformerSentenceEncoder,
        max_length: int = 128,
    ) -> None:
        queries: list[str] = []
        positives: list[str] = []
        negatives: list[str] = []
        for example in examples:
            queries.append(example.query_text)
            positives.append(example.positive_text)
            negatives.append(example.negative_text or example.positive_text)
        if not queries:
            raise ValueError("TextTripletDataset: empty examples list")
        self._q = encoder.tokenize(queries, max_length=max_length)
        self._p = encoder.tokenize(positives, max_length=max_length)
        self._n = encoder.tokenize(negatives, max_length=max_length)
        self._n_examples = len(queries)

    def __len__(self) -> int:
        return self._n_examples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "q_input_ids": self._q["input_ids"][index],
            "q_attention_mask": self._q["attention_mask"][index],
            "p_input_ids": self._p["input_ids"][index],
            "p_attention_mask": self._p["attention_mask"][index],
            "n_input_ids": self._n["input_ids"][index],
            "n_attention_mask": self._n["attention_mask"][index],
        }


# ---------------------------------------------------------------------------
# Шедулер: linear warmup + cosine decay
# ---------------------------------------------------------------------------
def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Тренер
# ---------------------------------------------------------------------------
@dataclass
class JointTrainingHistory:
    epochs: list[dict[str, float]] = field(default_factory=list)
    eval: list[dict[str, Any]] = field(default_factory=list)


class JointTrainer:
    """Совместное обучение трансформер-энкодера и хэш-MLP."""

    def __init__(
        self,
        encoder: TransformerSentenceEncoder,
        hash_model: HashingMLP,
        config: JointTrainingConfig,
        device: str | torch.device,
    ) -> None:
        self.encoder = encoder
        self.hash_model = hash_model
        self.config = config
        self.device = torch.device(device)
        self.encoder.set_trainable(True)
        self.hash_model.to(self.device)
        self.hash_model.train()

    # ------------------------------- internals ------------------------------
    def _hamming_proxy(self, left: torch.Tensor, right: torch.Tensor, code_bits: int) -> torch.Tensor:
        return 0.5 * (code_bits - (left * right).sum(dim=-1))

    def _triplet_loss(self, anchor, positive, negative) -> torch.Tensor:
        bits = int(anchor.shape[-1])
        d_pos = self._hamming_proxy(anchor, positive, bits)
        d_neg = self._hamming_proxy(anchor, negative, bits)
        return torch.relu(d_pos - d_neg + self.config.margin).mean()

    def _quantization_loss(self, relaxed: torch.Tensor) -> torch.Tensor:
        return (1.0 - relaxed.abs()).pow(2).mean()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        encoder_params = [p for p in self.encoder.model.parameters() if p.requires_grad]
        hash_params = [p for p in self.hash_model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.config.encoder_lr, "name": "encoder"},
                {"params": hash_params, "lr": self.config.hash_lr, "name": "hash"},
            ],
            weight_decay=self.config.weight_decay,
        )

    def _forward_branch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        dense = self.encoder.encode_torch(input_ids=input_ids, attention_mask=attention_mask)
        return self.hash_model(dense, temperature=self.config.temperature).relaxed

    # -------------------------------- API -----------------------------------
    def fit(
        self,
        dataset: TextTripletDataset,
        *,
        eval_callback: Callable[[int], dict[str, Any]] | None = None,
        checkpoint_dir: Path | None = None,
    ) -> JointTrainingHistory:
        torch.manual_seed(self.config.seed)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and self.device.type == "cuda",
            drop_last=False,
        )
        total_steps = max(1, len(loader) * self.config.epochs)
        warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))

        optimizer = self._build_optimizer()
        scheduler = build_warmup_cosine_scheduler(
            optimizer, total_steps=total_steps, warmup_steps=warmup_steps
        )

        amp_enabled = self.config.mixed_precision and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        history = JointTrainingHistory()
        global_step = 0
        for epoch in range(1, self.config.epochs + 1):
            epoch_started = time.perf_counter()
            self.encoder.set_trainable(True)
            self.hash_model.train()
            running = {"loss": 0.0, "triplet": 0.0, "quant": 0.0, "batches": 0}
            for batch in loader:
                global_step += 1
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                    q_relaxed = self._forward_branch(batch["q_input_ids"], batch["q_attention_mask"])
                    p_relaxed = self._forward_branch(batch["p_input_ids"], batch["p_attention_mask"])
                    n_relaxed = self._forward_branch(batch["n_input_ids"], batch["n_attention_mask"])
                    triplet_loss = self._triplet_loss(q_relaxed, p_relaxed, n_relaxed)
                    quant_loss = (
                        self._quantization_loss(q_relaxed)
                        + self._quantization_loss(p_relaxed)
                        + self._quantization_loss(n_relaxed)
                    ) / 3.0
                    loss = triplet_loss + self.config.quantization_weight * quant_loss

                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.model.parameters()) + list(self.hash_model.parameters()),
                        self.config.gradient_clip_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.model.parameters()) + list(self.hash_model.parameters()),
                        self.config.gradient_clip_norm,
                    )
                    optimizer.step()
                scheduler.step()

                running["loss"] += float(loss.item())
                running["triplet"] += float(triplet_loss.item())
                running["quant"] += float(quant_loss.item())
                running["batches"] += 1

                if global_step % self.config.log_every_n_steps == 0:
                    lr_groups = {g["name"]: g["lr"] for g in optimizer.param_groups}
                    print(
                        f"  step {global_step}/{total_steps}  loss={running['loss']/running['batches']:.4f}"
                        f"  triplet={running['triplet']/running['batches']:.4f}"
                        f"  quant={running['quant']/running['batches']:.4f}"
                        f"  lr_enc={lr_groups.get('encoder', 0):.2e}  lr_hash={lr_groups.get('hash', 0):.2e}",
                        flush=True,
                    )

            n = max(1, running["batches"])
            epoch_summary = {
                "epoch": epoch,
                "loss": running["loss"] / n,
                "triplet_loss": running["triplet"] / n,
                "quantization_loss": running["quant"] / n,
                "elapsed_s": round(time.perf_counter() - epoch_started, 1),
            }
            history.epochs.append(epoch_summary)
            print(
                f"[epoch {epoch}/{self.config.epochs}] loss={epoch_summary['loss']:.4f}  "
                f"triplet={epoch_summary['triplet_loss']:.4f}  quant={epoch_summary['quantization_loss']:.4f}  "
                f"({epoch_summary['elapsed_s']}s)",
                flush=True,
            )

            do_eval = eval_callback is not None and (
                epoch % self.config.eval_every_n_epochs == 0 or epoch == self.config.epochs
            )
            if do_eval:
                eval_metrics = eval_callback(epoch)
                eval_metrics["epoch"] = epoch
                history.eval.append(eval_metrics)

            if checkpoint_dir is not None and (
                epoch % self.config.save_every_n_epochs == 0 or epoch == self.config.epochs
            ):
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(checkpoint_dir / f"checkpoint_epoch{epoch}", optimizer, scheduler)
        return history

    # ------------------------------ persistence -----------------------------
    def save_checkpoint(
        self,
        path: Path,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.model.state_dict(), path / "encoder.pt")
        torch.save(self.hash_model.state_dict(), path / "hash_model.pt")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), path / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), path / "scheduler.pt")
        meta = {
            "encoder_model_name": self.encoder.config.model_name,
            "embedding_dim": self.encoder.embedding_dim,
            "hash_config": self.hash_model.config.to_dict(),
            "training_config": self.config.to_dict(),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
