from __future__ import annotations

import argparse
import random
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from chineseeeg2_littleprince.data import EEGTextDataset, collate_eeg_text
from chineseeeg2_littleprince.models import TemporalConvEEGEncoder


DEFAULTS = {
    "batch_size": 16,
    "max_samples": 1300,
    "num_workers": 0,
    "normalize_eeg": True,
    "seed": 42,
    "epochs": 5,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "val_fraction": 0.1,
    "contrastive_temperature": 0.07,
}

# 加载配置文件
def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def nested_get(config: dict[str, Any], *keys: str) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def resolve_manifest_path(manifest: str | Path, config_path: Path | None) -> Path:
    path = Path(manifest)
    if path.is_absolute():
        return path

    candidates = [Path.cwd() / path]
    if config_path is not None:
        candidates.extend([config_path.parent / path, config_path.parent.parent / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def split_indices(n_items: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n_items))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(n_items * val_fraction)))
    return indices[n_val:], indices[:n_val]


def cosine_mean(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(pred, target, dim=-1).mean()


def contrastive_logits(eeg_embedding: torch.Tensor, text_embedding: torch.Tensor, temperature: float) -> torch.Tensor:
    eeg_embedding = F.normalize(eeg_embedding, dim=-1)
    text_embedding = F.normalize(text_embedding, dim=-1)
    return eeg_embedding @ text_embedding.T / temperature


def eeg_to_text_contrastive_loss(
    eeg_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = contrastive_logits(eeg_embedding, text_embedding, temperature)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, targets), logits


def retrieval_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    targets = torch.arange(logits.shape[0], device=logits.device)
    k = min(k, logits.shape[1])
    predictions = logits.topk(k, dim=1).indices
    return (predictions == targets.unsqueeze(1)).any(dim=1).float().mean()


def run_epoch(model, loader, optimizer, device: torch.device, temperature: float) -> tuple[float, float, float, float]:
    model.train()
    total_loss = 0.0
    total_cos = 0.0
    total_eeg_to_text_top1 = 0.0
    total_eeg_to_text_top10 = 0.0
    total = 0
    for batch in loader:
        eeg = batch["eeg"].to(device)
        label = batch["label"].to(device)
        mask = batch["mask"].to(device)

        pred = model(eeg, mask)
        loss, logits = eeg_to_text_contrastive_loss(pred, label, temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = eeg.shape[0]
        total += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_cos += float(cosine_mean(pred.detach(), label)) * batch_size
        total_eeg_to_text_top1 += float(retrieval_topk(logits.detach(), k=1)) * batch_size
        total_eeg_to_text_top10 += float(retrieval_topk(logits.detach(), k=10)) * batch_size
    return total_loss / total, total_cos / total, total_eeg_to_text_top1 / total, total_eeg_to_text_top10 / total


@torch.no_grad()
def evaluate(model, loader, device: torch.device, temperature: float) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_cos = 0.0
    total_eeg_to_text_top1 = 0.0
    total_eeg_to_text_top10 = 0.0
    total = 0
    for batch in loader:
        eeg = batch["eeg"].to(device)
        label = batch["label"].to(device)
        mask = batch["mask"].to(device)
        pred = model(eeg, mask)
        loss, logits = eeg_to_text_contrastive_loss(pred, label, temperature)
        batch_size = eeg.shape[0]
        total += batch_size
        total_loss += float(loss) * batch_size
        total_cos += float(cosine_mean(pred, label)) * batch_size
        total_eeg_to_text_top1 += float(retrieval_topk(logits, k=1)) * batch_size
        total_eeg_to_text_top10 += float(retrieval_topk(logits, k=10)) * batch_size
    return total_loss / total, total_cos / total, total_eeg_to_text_top1 / total, total_eeg_to_text_top10 / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--normalize-eeg", dest="normalize_eeg", action="store_true", default=None)
    parser.add_argument("--no-normalize-eeg", dest="normalize_eeg", action="store_false")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument(
        "--temperature",
        "--contrastive-temperature",
        dest="contrastive_temperature",
        type=float,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config.resolve() if args.config is not None else None
    config = load_config(config_path)
    manifest = coalesce(args.manifest, config.get("manifest"))
    if manifest is None:
        parser.error("one of --config with a manifest field or --manifest is required")

    seed = int(coalesce(args.seed, config.get("seed"), DEFAULTS["seed"]))
    batch_size = int(coalesce(args.batch_size, config.get("batch_size"), DEFAULTS["batch_size"]))
    max_samples = int(coalesce(args.max_samples, config.get("max_samples"), DEFAULTS["max_samples"]))
    num_workers = int(coalesce(args.num_workers, config.get("num_workers"), DEFAULTS["num_workers"]))
    normalize_eeg = bool(coalesce(args.normalize_eeg, config.get("normalize_eeg"), DEFAULTS["normalize_eeg"]))
    epochs = int(coalesce(args.epochs, nested_get(config, "train", "epochs"), config.get("epochs"), DEFAULTS["epochs"]))
    lr = float(
        coalesce(args.lr, nested_get(config, "train", "learning_rate"), config.get("lr"), DEFAULTS["learning_rate"])
    )
    weight_decay = float(
        coalesce(
            args.weight_decay,
            nested_get(config, "train", "weight_decay"),
            config.get("weight_decay"),
            DEFAULTS["weight_decay"],
        )
    )
    val_fraction = float(
        coalesce(
            args.val_fraction,
            nested_get(config, "train", "val_fraction"),
            config.get("val_fraction"),
            DEFAULTS["val_fraction"],
        )
    )
    contrastive_temperature = float(
        coalesce(
            args.contrastive_temperature,
            nested_get(config, "train", "contrastive_temperature"),
            nested_get(config, "train", "temperature"),
            config.get("contrastive_temperature"),
            config.get("temperature"),
            DEFAULTS["contrastive_temperature"],
        )
    )
    if contrastive_temperature <= 0:
        raise ValueError(f"contrastive_temperature must be positive, got {contrastive_temperature}")
    device_name = str(coalesce(args.device, config.get("device"), "cuda" if torch.cuda.is_available() else "cpu"))
    model_kwargs = dict(config.get("model", {}))

    torch.manual_seed(seed)
    manifest_path = resolve_manifest_path(manifest, config_path)
    dataset = EEGTextDataset(manifest_path, normalize_eeg=normalize_eeg)
    train_idx, val_idx = split_indices(len(dataset), val_fraction, seed)
    collate_fn = partial(collate_eeg_text, max_samples=max_samples)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    device = torch.device(device_name)
    model = TemporalConvEEGEncoder(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        train_loss, train_cos, train_e2t_top1, train_e2t_top10 = run_epoch(
            model, train_loader, optimizer, device, contrastive_temperature
        )
        val_loss, val_cos, val_e2t_top1, val_e2t_top10 = evaluate(model, val_loader, device, contrastive_temperature)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} train_cos={train_cos:.4f} "
            f"train_top1={train_e2t_top1:.4f} train_top10={train_e2t_top10:.4f} "
            f"val_loss={val_loss:.5f} val_cos={val_cos:.4f} "
            f"val_top1={val_e2t_top1:.4f} val_top10={val_e2t_top10:.4f}"
        )


if __name__ == "__main__":
    main()
