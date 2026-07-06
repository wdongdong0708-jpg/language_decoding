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
    "test_fraction": 0.1,
    "contrastive_temperature": 0.07,
    "unique_text_per_batch": True,
    "early_stopping_patience": 8,
    "checkpoint_metric": "val_full_top10",
    "checkpoint_path": "checkpoints/best.pt",
}

METRIC_MODES = {
    "val_loss": "min",
    "val_cos": "max",
    "val_top1": "max",
    "val_top10": "max",
    "val_full_top1": "max",
    "val_full_top10": "max",
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


def _fraction_count(n_items: int, fraction: float) -> int:
    if fraction <= 0:
        return 0
    return max(1, int(round(n_items * fraction)))


def split_indices_by_text(
    records: list[Any],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if val_fraction < 0 or test_fraction < 0:
        raise ValueError("val_fraction and test_fraction must be non-negative")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be less than 1")

    text_ids = sorted({record.text_embedding_idx for record in records})
    rng = random.Random(seed)
    rng.shuffle(text_ids)

    n_val = _fraction_count(len(text_ids), val_fraction)
    n_test = _fraction_count(len(text_ids), test_fraction)
    if n_val + n_test >= len(text_ids):
        raise ValueError("Not enough unique text_embedding_idx values for the requested split fractions")

    val_text_ids = set(text_ids[:n_val])
    test_text_ids = set(text_ids[n_val : n_val + n_test])
    train_text_ids = set(text_ids[n_val + n_test :])

    train_indices = []
    val_indices = []
    test_indices = []
    for index, record in enumerate(records):
        text_id = record.text_embedding_idx
        if text_id in train_text_ids:
            train_indices.append(index)
        elif text_id in val_text_ids:
            val_indices.append(index)
        elif text_id in test_text_ids:
            test_indices.append(index)

    return train_indices, val_indices, test_indices


class UniqueTextBatchSampler:
    """Yield full-dataset indices with at most one sample per text id in each batch."""

    def __init__(
        self,
        indices: list[int],
        text_embedding_indices: list[int],
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.indices = list(indices)
        self.text_embedding_indices = text_embedding_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

    def _make_batches(self, rng: random.Random | None = None) -> list[list[int]]:
        ordered_indices = list(self.indices)
        if self.shuffle and rng is not None:
            rng.shuffle(ordered_indices)

        grouped: dict[int, list[int]] = {}
        for index in ordered_indices:
            text_id = self.text_embedding_indices[index]
            grouped.setdefault(text_id, []).append(index)

        keys = list(grouped)
        if not self.shuffle:
            keys.sort()

        batches = []
        while True:
            active_keys = [key for key in keys if grouped[key]]
            if not active_keys:
                break
            if self.shuffle and rng is not None:
                rng.shuffle(active_keys)

            batch = []
            for key in active_keys:
                batch.append(grouped[key].pop(0))
                if len(batch) == self.batch_size:
                    batches.append(batch)
                    batch = []
            if batch and not self.drop_last:
                batches.append(batch)

        return batches

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        yield from self._make_batches(rng)

    def __len__(self) -> int:
        return len(self._make_batches())


def make_loader(
    dataset: EEGTextDataset,
    indices: list[int],
    batch_size: int,
    collate_fn,
    num_workers: int,
    shuffle: bool,
    unique_text_per_batch: bool,
    seed: int,
) -> DataLoader:
    if unique_text_per_batch:
        text_embedding_indices = [record.text_embedding_idx for record in dataset.records]
        return DataLoader(
            dataset,
            batch_sampler=UniqueTextBatchSampler(
                indices=indices,
                text_embedding_indices=text_embedding_indices,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
            ),
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def cosine_mean(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(pred, target, dim=-1).mean()


def contrastive_logits(eeg_embedding: torch.Tensor, text_embedding: torch.Tensor, temperature: float) -> torch.Tensor:
    eeg_embedding = F.normalize(eeg_embedding, dim=-1)
    text_embedding = F.normalize(text_embedding, dim=-1)
    return eeg_embedding @ text_embedding.T / temperature


def eeg_to_text_contrastive_loss(
    eeg_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    text_embedding_idx: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = contrastive_logits(eeg_embedding, text_embedding, temperature)
    positive_mask = text_embedding_idx.unsqueeze(1).eq(text_embedding_idx.unsqueeze(0))
    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = -(log_probs * positive_mask.to(log_probs.dtype)).sum(dim=1) / positive_counts
    return loss.mean(), logits


def retrieval_topk(logits: torch.Tensor, text_embedding_idx: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.shape[1])
    predictions = logits.topk(k, dim=1).indices
    positive_mask = text_embedding_idx.unsqueeze(1).eq(text_embedding_idx.unsqueeze(0))
    return positive_mask.gather(dim=1, index=predictions).any(dim=1).float().mean()


def _unique_text_candidates(
    text_embedding: torch.Tensor,
    text_embedding_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_by_text_id: dict[int, int] = {}
    for index, text_id in enumerate(text_embedding_idx.detach().cpu().tolist()):
        first_by_text_id.setdefault(int(text_id), index)

    ordered_text_ids = sorted(first_by_text_id)
    candidate_indices = torch.tensor(
        [first_by_text_id[text_id] for text_id in ordered_text_ids],
        dtype=torch.long,
        device=text_embedding.device,
    )
    return text_embedding.index_select(0, candidate_indices), torch.tensor(
        ordered_text_ids,
        dtype=text_embedding_idx.dtype,
        device=text_embedding_idx.device,
    )


def full_retrieval_topk(
    eeg_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    text_embedding_idx: torch.Tensor,
    k: int,
    chunk_size: int = 1024,
) -> torch.Tensor:
    text_candidates, candidate_text_ids = _unique_text_candidates(text_embedding, text_embedding_idx)
    eeg_embedding = F.normalize(eeg_embedding, dim=-1)
    text_candidates = F.normalize(text_candidates, dim=-1)
    k = min(k, text_candidates.shape[0])

    total_correct = 0.0
    total = 0
    for start in range(0, eeg_embedding.shape[0], chunk_size):
        stop = min(start + chunk_size, eeg_embedding.shape[0])
        logits = eeg_embedding[start:stop] @ text_candidates.T
        predictions = logits.topk(k, dim=1).indices
        query_text_ids = text_embedding_idx[start:stop]
        positive_mask = query_text_ids.unsqueeze(1).eq(candidate_text_ids.unsqueeze(0))
        total_correct += float(positive_mask.gather(dim=1, index=predictions).any(dim=1).float().sum())
        total += stop - start

    return torch.tensor(total_correct / max(total, 1), device=eeg_embedding.device)


def checkpoint_is_better(metric: float, best_metric: float | None, mode: str) -> bool:
    if best_metric is None:
        return True
    if mode == "min":
        return metric < best_metric
    if mode == "max":
        return metric > best_metric
    raise ValueError(f"Unsupported checkpoint metric mode: {mode}")


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
        text_embedding_idx = batch["text_embedding_idx"].to(device)

        pred = model(eeg, mask)
        loss, logits = eeg_to_text_contrastive_loss(pred, label, text_embedding_idx, temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = eeg.shape[0]
        total += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_cos += float(cosine_mean(pred.detach(), label)) * batch_size
        total_eeg_to_text_top1 += float(retrieval_topk(logits.detach(), text_embedding_idx, k=1)) * batch_size
        total_eeg_to_text_top10 += float(retrieval_topk(logits.detach(), text_embedding_idx, k=10)) * batch_size
    return total_loss / total, total_cos / total, total_eeg_to_text_top1 / total, total_eeg_to_text_top10 / total


@torch.no_grad()
def evaluate(model, loader, device: torch.device, temperature: float) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_cos = 0.0
    total_eeg_to_text_top1 = 0.0
    total_eeg_to_text_top10 = 0.0
    total = 0
    all_pred = []
    all_label = []
    all_text_embedding_idx = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        label = batch["label"].to(device)
        mask = batch["mask"].to(device)
        text_embedding_idx = batch["text_embedding_idx"].to(device)
        pred = model(eeg, mask)
        loss, logits = eeg_to_text_contrastive_loss(pred, label, text_embedding_idx, temperature)
        batch_size = eeg.shape[0]
        total += batch_size
        total_loss += float(loss) * batch_size
        total_cos += float(cosine_mean(pred, label)) * batch_size
        total_eeg_to_text_top1 += float(retrieval_topk(logits, text_embedding_idx, k=1)) * batch_size
        total_eeg_to_text_top10 += float(retrieval_topk(logits, text_embedding_idx, k=10)) * batch_size
        all_pred.append(pred.detach())
        all_label.append(label.detach())
        all_text_embedding_idx.append(text_embedding_idx.detach())

    pred_all = torch.cat(all_pred, dim=0)
    label_all = torch.cat(all_label, dim=0)
    text_embedding_idx_all = torch.cat(all_text_embedding_idx, dim=0)
    return {
        "loss": total_loss / total,
        "cos": total_cos / total,
        "top1": total_eeg_to_text_top1 / total,
        "top10": total_eeg_to_text_top10 / total,
        "full_top1": float(full_retrieval_topk(pred_all, label_all, text_embedding_idx_all, k=1)),
        "full_top10": float(full_retrieval_topk(pred_all, label_all, text_embedding_idx_all, k=10)),
    }


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
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument(
        "--temperature",
        "--contrastive-temperature",
        dest="contrastive_temperature",
        type=float,
        default=None,
    )
    parser.add_argument("--unique-text-per-batch", dest="unique_text_per_batch", action="store_true", default=None)
    parser.add_argument("--allow-duplicate-text-per-batch", dest="unique_text_per_batch", action="store_false")
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--checkpoint-metric", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
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
    test_fraction = float(
        coalesce(
            args.test_fraction,
            nested_get(config, "train", "test_fraction"),
            config.get("test_fraction"),
            DEFAULTS["test_fraction"],
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
    unique_text_per_batch = bool(
        coalesce(
            args.unique_text_per_batch,
            nested_get(config, "train", "unique_text_per_batch"),
            config.get("unique_text_per_batch"),
            DEFAULTS["unique_text_per_batch"],
        )
    )
    early_stopping_patience = int(
        coalesce(
            args.early_stopping_patience,
            nested_get(config, "train", "early_stopping_patience"),
            config.get("early_stopping_patience"),
            DEFAULTS["early_stopping_patience"],
        )
    )
    checkpoint_metric = str(
        coalesce(
            args.checkpoint_metric,
            nested_get(config, "train", "checkpoint_metric"),
            config.get("checkpoint_metric"),
            DEFAULTS["checkpoint_metric"],
        )
    )
    if checkpoint_metric not in METRIC_MODES:
        raise ValueError(f"checkpoint_metric must be one of {sorted(METRIC_MODES)}, got {checkpoint_metric!r}")
    checkpoint_path = Path(
        coalesce(
            args.checkpoint_path,
            nested_get(config, "train", "checkpoint_path"),
            config.get("checkpoint_path"),
            DEFAULTS["checkpoint_path"],
        )
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    device_name = str(coalesce(args.device, config.get("device"), "cuda" if torch.cuda.is_available() else "cpu"))
    model_kwargs = dict(config.get("model", {}))

    torch.manual_seed(seed)
    manifest_path = resolve_manifest_path(manifest, config_path)
    dataset = EEGTextDataset(manifest_path, normalize_eeg=normalize_eeg)
    train_idx, val_idx, test_idx = split_indices_by_text(dataset.records, val_fraction, test_fraction, seed)
    collate_fn = partial(collate_eeg_text, max_samples=max_samples)

    train_loader = make_loader(
        dataset,
        train_idx,
        batch_size,
        collate_fn,
        num_workers,
        shuffle=True,
        unique_text_per_batch=unique_text_per_batch,
        seed=seed,
    )
    val_loader = make_loader(
        dataset,
        val_idx,
        batch_size,
        collate_fn,
        num_workers,
        shuffle=False,
        unique_text_per_batch=unique_text_per_batch,
        seed=seed + 1,
    )
    test_loader = (
        make_loader(
            dataset,
            test_idx,
            batch_size,
            collate_fn,
            num_workers,
            shuffle=False,
            unique_text_per_batch=unique_text_per_batch,
            seed=seed + 2,
        )
        if test_idx
        else None
    )

    print(
        f"split_rows train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"unique_text_per_batch={unique_text_per_batch}"
    )

    device = torch.device(device_name)
    model = TemporalConvEEGEncoder(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_mode = METRIC_MODES[checkpoint_metric]
    best_metric = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_cos, train_e2t_top1, train_e2t_top10 = run_epoch(
            model, train_loader, optimizer, device, contrastive_temperature
        )
        val_metrics = evaluate(model, val_loader, device, contrastive_temperature)
        current_metrics = {
            "val_loss": val_metrics["loss"],
            "val_cos": val_metrics["cos"],
            "val_top1": val_metrics["top1"],
            "val_top10": val_metrics["top10"],
            "val_full_top1": val_metrics["full_top1"],
            "val_full_top10": val_metrics["full_top10"],
        }
        metric_value = current_metrics[checkpoint_metric]
        improved = checkpoint_is_better(metric_value, best_metric, checkpoint_mode)
        if improved:
            best_metric = metric_value
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "checkpoint_metric": checkpoint_metric,
                    "checkpoint_metric_value": metric_value,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": current_metrics,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} train_cos={train_cos:.4f} "
            f"train_top1={train_e2t_top1:.4f} train_top10={train_e2t_top10:.4f} "
            f"val_loss={val_metrics['loss']:.5f} val_cos={val_metrics['cos']:.4f} "
            f"val_top1={val_metrics['top1']:.4f} val_top10={val_metrics['top10']:.4f} "
            f"val_full_top1={val_metrics['full_top1']:.4f} val_full_top10={val_metrics['full_top10']:.4f} "
            f"best_{checkpoint_metric}={best_metric:.4f} best_epoch={best_epoch}"
        )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"early_stopping epoch={epoch:03d} best_epoch={best_epoch:03d} "
                f"best_{checkpoint_metric}={best_metric:.4f}"
            )
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"loaded_best_checkpoint path={checkpoint_path} epoch={checkpoint['epoch']} "
        f"{checkpoint_metric}={checkpoint['checkpoint_metric_value']:.4f}"
    )

    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, device, contrastive_temperature)
        print(
            f"test_loss={test_metrics['loss']:.5f} test_cos={test_metrics['cos']:.4f} "
            f"test_top1={test_metrics['top1']:.4f} test_top10={test_metrics['top10']:.4f} "
            f"test_full_top1={test_metrics['full_top1']:.4f} test_full_top10={test_metrics['full_top10']:.4f}"
        )


if __name__ == "__main__":
    main()
