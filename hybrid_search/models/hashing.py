from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from ..core.config import HashingModelConfig


class StraightThroughSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.where(input_tensor >= 0, torch.ones_like(input_tensor), -torch.ones_like(input_tensor))

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.clamp(-1.0, 1.0)


def ste_sign(input_tensor: torch.Tensor) -> torch.Tensor:
    return StraightThroughSign.apply(input_tensor)


def _activation(name: str) -> nn.Module:
    mapping = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation: {name}")
    return mapping[name]()


@dataclass
class HashingOutput:
    logits: torch.Tensor
    relaxed: torch.Tensor
    binary: torch.Tensor


class HashingMLP(nn.Module):
    def __init__(self, config: HashingModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dims = tuple(config.hidden_dims)
        layers = []
        in_features = config.input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_features = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = nn.Linear(in_features, config.code_bits)

    def forward(self, embeddings: torch.Tensor, temperature: float = 1.0) -> HashingOutput:
        hidden = self.backbone(embeddings)
        logits = self.output(hidden)
        relaxed = torch.tanh(logits * temperature)
        binary = ste_sign(logits)
        return HashingOutput(logits=logits, relaxed=relaxed, binary=binary)

    @torch.no_grad()
    def encode_embeddings(
        self,
        embeddings: np.ndarray | torch.Tensor,
        batch_size: int = 512,
        device: str | torch.device = "cpu",
    ) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            tensor = torch.from_numpy(embeddings.astype(np.float32, copy=False))
        else:
            tensor = embeddings.detach().float().cpu()
        self.eval()
        device_obj = torch.device(device)
        self.to(device_obj)
        encoded = []
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(device_obj)
            output = self.forward(batch)
            encoded.append(output.binary.detach().cpu().numpy().astype(np.int8))
        return np.vstack(encoded)


def binary_codes_to_bits(codes: np.ndarray) -> np.ndarray:
    return (codes > 0).astype(np.uint8)


def pack_binary_codes(codes: np.ndarray) -> np.ndarray:
    bit_matrix = binary_codes_to_bits(codes)
    return np.packbits(bit_matrix, axis=1, bitorder="little")


def unpack_binary_codes(packed_codes: np.ndarray, code_bits: int) -> np.ndarray:
    unpacked = np.unpackbits(packed_codes, axis=1, bitorder="little")
    return unpacked[:, :code_bits].astype(np.int8) * 2 - 1


def build_hash_model(config: HashingModelConfig) -> HashingMLP:
    return HashingMLP(config)


def save_hash_checkpoint(path: str | Path, model: HashingMLP, extra: Dict[str, Any] | None = None) -> None:
    payload = {
        "model_config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, Path(path))


def load_hash_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[HashingMLP, Dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device)
    config_payload = payload["model_config"]
    config = HashingModelConfig(
        input_dim=int(config_payload["input_dim"]),
        code_bits=int(config_payload["code_bits"]),
        hidden_dims=tuple(config_payload["hidden_dims"]),
        activation=str(config_payload["activation"]),
        dropout=float(config_payload["dropout"]),
        use_layer_norm=bool(config_payload["use_layer_norm"]),
    )
    model = HashingMLP(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, payload.get("extra", {})
