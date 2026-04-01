from __future__ import annotations

from dataclasses import asdict
from typing import Protocol, Sequence

import numpy as np
import torch

from ..core.config import EncoderConfig
from ..core.preprocessing import TextPreprocessor
from ..core.utils import batched, l2_normalize, stable_hash_int


class BaseTextEncoder(Protocol):
    @property
    def embedding_dim(self) -> int:
        ...

    @property
    def config(self) -> EncoderConfig:
        ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


class HashingTextEncoder:
    def __init__(
        self,
        config: EncoderConfig | None = None,
        preprocessor: TextPreprocessor | None = None,
    ) -> None:
        self._config = config or EncoderConfig()
        self.preprocessor = preprocessor or TextPreprocessor()

    @property
    def embedding_dim(self) -> int:
        return self._config.embedding_dim

    @property
    def config(self) -> EncoderConfig:
        return self._config

    def _token_features(self, token: str) -> list[str]:
        features = [token]
        if len(token) > 3:
            features.extend(token[index : index + 3] for index in range(len(token) - 2))
        return features

    def _encode_single(self, text: str) -> np.ndarray:
        vector = np.zeros(self.embedding_dim, dtype=np.float32)
        tokens = self.preprocessor.tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            for feature in self._token_features(token):
                idx = stable_hash_int(feature, salt=str(self._config.seed)) % self.embedding_dim
                sign = -1.0 if stable_hash_int(feature, salt="sign") % 2 else 1.0
                vector[idx] += sign
        vector /= max(len(tokens), 1)
        if self._config.normalize_embeddings:
            vector = l2_normalize(vector.reshape(1, -1))[0]
        return vector.astype(np.float32, copy=False)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        normalized = self.preprocessor.normalize_batch(texts)
        matrix = np.stack([self._encode_single(text) for text in normalized], axis=0)
        return matrix.astype(np.float32, copy=False)


class TransformerSentenceEncoder:
    def __init__(
        self,
        config: EncoderConfig | None = None,
        preprocessor: TextPreprocessor | None = None,
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. Install the optional dependency to use this backend."
            ) from exc

        self._config = config or EncoderConfig(backend="transformers")
        self.preprocessor = preprocessor or TextPreprocessor()
        self.device = torch.device(self._config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self._config.model_name)
        self.model = AutoModel.from_pretrained(self._config.model_name).to(self.device)
        self.model.eval()
        self._embedding_dim = int(self.model.config.hidden_size)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def config(self) -> EncoderConfig:
        return self._config

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1e-9)
        return summed / counts

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        normalized = self.preprocessor.normalize_batch(texts)
        batches = []
        for batch in batched(normalized, self._config.batch_size):
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.preprocessor.config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            outputs = self.model(**encoded)
            pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            vector_batch = pooled.detach().cpu().numpy().astype(np.float32)
            if self._config.normalize_embeddings:
                vector_batch = l2_normalize(vector_batch)
            batches.append(vector_batch)
        return np.vstack(batches)


def build_encoder(
    config: EncoderConfig | None = None,
    preprocessor: TextPreprocessor | None = None,
) -> BaseTextEncoder:
    resolved = config or EncoderConfig()
    if resolved.backend == "hashing":
        return HashingTextEncoder(resolved, preprocessor=preprocessor)
    if resolved.backend == "transformers":
        return TransformerSentenceEncoder(resolved, preprocessor=preprocessor)
    raise ValueError(f"Unsupported encoder backend: {resolved.backend}")
