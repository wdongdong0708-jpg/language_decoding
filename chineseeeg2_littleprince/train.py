from __future__ import annotations

import argparse
import math
import random
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from chineseeeg2_littleprince.data import EEGTextDataset, collate_eeg_text, split_indices_by_group
from chineseeeg2_littleprince.models import build_eeg_encoder
from chineseeeg2_littleprince.retrieval import (
    compute_full_retrieval_metrics,
    full_retrieval_topk,
)


DEFAULTS = {
    "batch_size": 16,
    "max_samples": 1300,
    "time_fit_mode": "crop_pad",
    "num_workers": 0,
    "normalize_eeg": True,
    "seed": 42,
    "epochs": 5,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "val_fraction": 0.1,
    "test_fraction": 0.1,
    "contrastive_temperature": 0.07,
    "unique_target_per_batch": True,
    "train_batch_mode": "unique_target",
    "targets_per_batch": 32,
    "views_per_target": 4,
    "view_groups_per_target_per_epoch": 1,
    "drop_last_target_batch": True,
    "min_train_batch_size": 32,
    "early_stopping_patience": 8,
    "checkpoint_metric": "val_full_macro_top10",
    "checkpoint_path": "checkpoints/best.pt",
}

METRIC_MODES = {
    "val_loss": "min",
    "val_cos": "max",
    "val_top1": "max",
    "val_top10": "max",
    "val_full_top1": "max",
    "val_full_top10": "max",
    "val_full_macro_top1": "max",
    "val_full_macro_top10": "max",
    "val_full_mean_rank": "min",
    "val_full_median_rank": "min",
    "val_full_macro_mean_rank": "min",
    "val_full_macro_median_rank": "min",
    "val_full_instance_top1": "max",
    "val_full_instance_top10": "max",
    "val_full_instance_mean_rank": "min",
    "val_full_instance_median_rank": "min",
    "val_top250_top1": "max",
    "val_top250_top10": "max",
    "val_top250_macro_top1": "max",
    "val_top250_macro_top10": "max",
    "val_top250_mean_rank": "min",
    "val_top250_median_rank": "min",
    "val_top250_macro_mean_rank": "min",
    "val_top250_macro_median_rank": "min",
    "val_top250_instance_top1": "max",
    "val_top250_instance_top10": "max",
    "val_top250_instance_mean_rank": "min",
    "val_top250_instance_median_rank": "min",
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


split_indices_by_text = split_indices_by_group


class UniqueTargetBatchSampler:
    """Yield batches with at most one sample per canonical target."""

    def __init__(
        self,
        indices: list[int],
        target_ids: list[int],
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
        min_batch_size: int = 1,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not 1 <= min_batch_size <= batch_size:
            raise ValueError(
                f"min_batch_size must be in [1, {batch_size}], got {min_batch_size}"
            )
        self.indices = list(indices)
        self.target_ids = target_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.min_batch_size = min_batch_size
        self._epoch = 0

    def _make_batches(self, rng: random.Random | None = None) -> list[list[int]]:
        ordered_indices = list(self.indices)
        if self.shuffle and rng is not None:
            rng.shuffle(ordered_indices)

        grouped: dict[int, list[int]] = {}
        for index in ordered_indices:
            target_id = self.target_ids[index]
            grouped.setdefault(target_id, []).append(index)

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

        batches = [batch for batch in batches if len(batch) >= self.min_batch_size]
        if self.shuffle and rng is not None:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        yield from self._make_batches(rng)

    def __len__(self) -> int:
        return len(self._make_batches())

    @property
    def retained_samples(self) -> int:
        return sum(len(batch) for batch in self._make_batches())

    @property
    def dropped_samples(self) -> int:
        return len(self.indices) - self.retained_samples


UniqueTextBatchSampler = UniqueTargetBatchSampler


class MultiPositiveTargetBatchSampler:
    """Yield batches containing several distinct EEG views for each target."""

    def __init__(
        self,
        indices: list[int],
        target_ids: list[int],
        targets_per_batch: int,
        views_per_target: int,
        shuffle: bool,
        seed: int,
        view_groups_per_target_per_epoch: int = 1,
        drop_last: bool = True,
    ):
        if targets_per_batch <= 0:
            raise ValueError(
                f"targets_per_batch must be positive, got {targets_per_batch}"
            )
        if views_per_target <= 1:
            raise ValueError(
                f"views_per_target must be greater than 1, got {views_per_target}"
            )
        if view_groups_per_target_per_epoch <= 0:
            raise ValueError(
                "view_groups_per_target_per_epoch must be positive, got "
                f"{view_groups_per_target_per_epoch}"
            )
        self.indices = list(indices)
        self.target_ids = target_ids
        self.targets_per_batch = targets_per_batch
        self.views_per_target = views_per_target
        self.shuffle = shuffle
        self.seed = seed
        self.view_groups_per_target_per_epoch = view_groups_per_target_per_epoch
        self.drop_last = drop_last
        self._epoch = 0

        grouped: dict[int, list[int]] = {}
        for index in self.indices:
            grouped.setdefault(target_ids[index], []).append(index)
        too_small = {
            target_id: len(target_indices)
            for target_id, target_indices in grouped.items()
            if len(target_indices) < views_per_target
        }
        if too_small:
            preview = sorted(too_small.items())[:10]
            raise ValueError(
                f"{len(too_small)} targets have fewer than {views_per_target} views: "
                f"{preview}"
            )
        self.grouped_indices = {
            target_id: sorted(target_indices)
            for target_id, target_indices in grouped.items()
        }

    def _target_views(self, target_id: int, group_index: int) -> list[int]:
        indices = self.grouped_indices[target_id]
        n_views = len(indices)
        cycle_span = math.lcm(n_views, self.views_per_target)
        consumed = group_index * self.views_per_target
        cycle = consumed // cycle_span
        offset = consumed % n_views
        permutation = list(indices)
        random.Random(f"{self.seed}:{target_id}:{cycle}").shuffle(permutation)
        return [
            permutation[(offset + view_index) % n_views]
            for view_index in range(self.views_per_target)
        ]

    def _make_batches(self, epoch: int) -> list[list[int]]:
        ordered_targets = sorted(self.grouped_indices)
        batches = []
        for view_round in range(self.view_groups_per_target_per_epoch):
            round_targets = list(ordered_targets)
            rng = random.Random(
                self.seed
                + epoch * self.view_groups_per_target_per_epoch
                + view_round
            )
            if self.shuffle:
                rng.shuffle(round_targets)
            for start in range(0, len(round_targets), self.targets_per_batch):
                batch_targets = round_targets[start : start + self.targets_per_batch]
                if len(batch_targets) < self.targets_per_batch and self.drop_last:
                    continue
                group_index = (
                    epoch * self.view_groups_per_target_per_epoch + view_round
                )
                batch = [
                    index
                    for target_id in batch_targets
                    for index in self._target_views(target_id, group_index)
                ]
                if self.shuffle:
                    rng.shuffle(batch)
                batches.append(batch)
        return batches

    def __iter__(self):
        epoch = self._epoch
        self._epoch += 1
        yield from self._make_batches(epoch)

    def __len__(self) -> int:
        targets = len(self.grouped_indices)
        batches_per_round = (
            targets // self.targets_per_batch
            if self.drop_last
            else math.ceil(targets / self.targets_per_batch)
        )
        return batches_per_round * self.view_groups_per_target_per_epoch

    @property
    def samples_per_epoch(self) -> int:
        return sum(len(batch) for batch in self._make_batches(epoch=0))

    @property
    def unique_samples_first_epoch(self) -> int:
        return len(
            {
                index
                for batch in self._make_batches(epoch=0)
                for index in batch
            }
        )

    @property
    def target_groups_per_epoch(self) -> int:
        return sum(
            len(batch) // self.views_per_target
            for batch in self._make_batches(epoch=0)
        )


def make_loader(
    dataset: EEGTextDataset,
    indices: list[int],
    batch_size: int,
    collate_fn,
    num_workers: int,
    shuffle: bool,
    unique_target_per_batch: bool,
    seed: int,
    min_batch_size: int = 1,
) -> DataLoader:
    if unique_target_per_batch:
        target_ids = [record.target_id for record in dataset.records]
        return DataLoader(
            dataset,
            batch_sampler=UniqueTargetBatchSampler(
                indices=indices,
                target_ids=target_ids,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                min_batch_size=min_batch_size,
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


def text_embedding_for_similarity(model, text_embedding: torch.Tensor) -> torch.Tensor:
    """Apply an encoder-specific text projection when one is available."""

    projector = getattr(model, "project_text_embedding", None)
    if projector is None:
        return text_embedding
    return projector(text_embedding)


def contrastive_logits(eeg_embedding: torch.Tensor, text_embedding: torch.Tensor, temperature: float) -> torch.Tensor:
    eeg_embedding = F.normalize(eeg_embedding, dim=-1)
    text_embedding = F.normalize(text_embedding, dim=-1)
    return eeg_embedding @ text_embedding.T / temperature


def eeg_to_text_contrastive_loss(
    eeg_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    target_id: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = contrastive_logits(eeg_embedding, text_embedding, temperature)
    positive_mask = target_id.unsqueeze(1).eq(target_id.unsqueeze(0))
    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = -(log_probs * positive_mask.to(log_probs.dtype)).sum(dim=1) / positive_counts
    return loss.mean(), logits


def retrieval_topk(logits: torch.Tensor, target_id: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.shape[1])
    predictions = logits.topk(k, dim=1).indices
    positive_mask = target_id.unsqueeze(1).eq(target_id.unsqueeze(0))
    return positive_mask.gather(dim=1, index=predictions).any(dim=1).float().mean()


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
        target_id = batch["target_id"].to(device)
        subject_id = batch.get("subject_id")
        if subject_id is not None:
            subject_id = subject_id.to(device)

        pred = model(eeg, mask, subject_id=subject_id)
        projected_label = text_embedding_for_similarity(model, label)
        loss, logits = eeg_to_text_contrastive_loss(
            pred, projected_label, target_id, temperature
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = eeg.shape[0]
        total += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_cos += float(
            cosine_mean(pred.detach(), projected_label.detach())
        ) * batch_size
        total_eeg_to_text_top1 += float(retrieval_topk(logits.detach(), target_id, k=1)) * batch_size
        total_eeg_to_text_top10 += float(retrieval_topk(logits.detach(), target_id, k=10)) * batch_size
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
    all_target_id = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        label = batch["label"].to(device)
        mask = batch["mask"].to(device)
        target_id = batch["target_id"].to(device)
        subject_id = batch.get("subject_id")
        if subject_id is not None:
            subject_id = subject_id.to(device)
        pred = model(eeg, mask, subject_id=subject_id)
        projected_label = text_embedding_for_similarity(model, label)
        loss, logits = eeg_to_text_contrastive_loss(
            pred, projected_label, target_id, temperature
        )
        batch_size = eeg.shape[0]
        total += batch_size
        total_loss += float(loss) * batch_size
        total_cos += float(cosine_mean(pred, projected_label)) * batch_size
        total_eeg_to_text_top1 += float(retrieval_topk(logits, target_id, k=1)) * batch_size
        total_eeg_to_text_top10 += float(retrieval_topk(logits, target_id, k=10)) * batch_size
        all_pred.append(pred.detach())
        all_label.append(projected_label.detach())
        all_target_id.append(target_id.detach())

    pred_all = torch.cat(all_pred, dim=0)
    label_all = torch.cat(all_label, dim=0)
    target_id_all = torch.cat(all_target_id, dim=0)
    metrics = {
        "loss": total_loss / total,
        "cos": total_cos / total,
        "top1": total_eeg_to_text_top1 / total,
        "top10": total_eeg_to_text_top10 / total,
    }
    for prefix, candidate_limit in [("full", None), ("top250", 250)]:
        retrieval_metrics = compute_full_retrieval_metrics(
            pred_all,
            label_all,
            target_id_all,
            candidate_limit=candidate_limit,
        )
        metrics.update({f"{prefix}_{name}": value for name, value in retrieval_metrics.items()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--time-fit-mode",
        choices=("crop_pad", "resample"),
        default=None,
    )
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
    parser.add_argument(
        "--unique-target-per-batch",
        "--unique-text-per-batch",
        dest="unique_target_per_batch",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--allow-duplicate-target-per-batch",
        "--allow-duplicate-text-per-batch",
        dest="unique_target_per_batch",
        action="store_false",
    )
    parser.add_argument("--min-train-batch-size", type=int, default=None)
    parser.add_argument(
        "--train-batch-mode",
        choices=("unique_target", "multi_positive"),
        default=None,
    )
    parser.add_argument("--targets-per-batch", type=int, default=None)
    parser.add_argument("--views-per-target", type=int, default=None)
    parser.add_argument(
        "--view-groups-per-target-per-epoch", type=int, default=None
    )
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
    time_fit_mode = str(
        coalesce(
            args.time_fit_mode,
            config.get("time_fit_mode"),
            DEFAULTS["time_fit_mode"],
        )
    )
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
    unique_target_per_batch = bool(
        coalesce(
            args.unique_target_per_batch,
            nested_get(config, "train", "unique_target_per_batch"),
            nested_get(config, "train", "unique_text_per_batch"),
            config.get("unique_target_per_batch"),
            config.get("unique_text_per_batch"),
            DEFAULTS["unique_target_per_batch"],
        )
    )
    min_train_batch_size = int(
        coalesce(
            args.min_train_batch_size,
            nested_get(config, "train", "min_train_batch_size"),
            config.get("min_train_batch_size"),
            DEFAULTS["min_train_batch_size"],
        )
    )
    if not 1 <= min_train_batch_size <= batch_size:
        raise ValueError(
            f"min_train_batch_size must be in [1, {batch_size}], "
            f"got {min_train_batch_size}"
        )
    train_batch_mode = str(
        coalesce(
            args.train_batch_mode,
            nested_get(config, "train", "batch_mode"),
            config.get("train_batch_mode"),
            DEFAULTS["train_batch_mode"],
        )
    )
    if train_batch_mode not in {"unique_target", "multi_positive"}:
        raise ValueError(
            "train_batch_mode must be 'unique_target' or 'multi_positive', "
            f"got {train_batch_mode!r}"
        )
    targets_per_batch = int(
        coalesce(
            args.targets_per_batch,
            nested_get(config, "train", "targets_per_batch"),
            config.get("targets_per_batch"),
            DEFAULTS["targets_per_batch"],
        )
    )
    views_per_target = int(
        coalesce(
            args.views_per_target,
            nested_get(config, "train", "views_per_target"),
            config.get("views_per_target"),
            DEFAULTS["views_per_target"],
        )
    )
    view_groups_per_target_per_epoch = int(
        coalesce(
            args.view_groups_per_target_per_epoch,
            nested_get(config, "train", "view_groups_per_target_per_epoch"),
            config.get("view_groups_per_target_per_epoch"),
            DEFAULTS["view_groups_per_target_per_epoch"],
        )
    )
    drop_last_target_batch = bool(
        coalesce(
            nested_get(config, "train", "drop_last_target_batch"),
            config.get("drop_last_target_batch"),
            DEFAULTS["drop_last_target_batch"],
        )
    )
    if train_batch_mode == "multi_positive":
        expected_batch_size = targets_per_batch * views_per_target
        if expected_batch_size != batch_size:
            raise ValueError(
                "multi_positive batching requires batch_size == "
                "targets_per_batch * views_per_target, got "
                f"{batch_size} != {targets_per_batch} * {views_per_target}"
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
    model_name = str(model_kwargs.pop("name", "temporal_conv"))

    torch.manual_seed(seed)
    manifest_path = resolve_manifest_path(manifest, config_path)
    subjects = config.get("subjects")
    dataset = EEGTextDataset(
        manifest_path,
        normalize_eeg=normalize_eeg,
        subjects=subjects,
    )
    subject_layers_enabled = bool(model_kwargs.get("subject_layers", False))
    if subject_layers_enabled:
        n_dataset_subjects = len(dataset.subject_to_id)
        configured_n_subjects = model_kwargs.get("n_subjects")
        if configured_n_subjects is None:
            model_kwargs["n_subjects"] = n_dataset_subjects
        elif int(configured_n_subjects) < n_dataset_subjects:
            raise ValueError(
                f"model.n_subjects={configured_n_subjects} is smaller than the "
                f"{n_dataset_subjects} subjects present in the manifest"
            )
        else:
            model_kwargs["n_subjects"] = int(configured_n_subjects)
    train_idx, val_idx, test_idx = split_indices_by_group(dataset.records, val_fraction, test_fraction, seed)
    if not train_idx or not val_idx:
        raise ValueError(
            f"Stable group split produced an empty required split: "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )
    collate_fn = partial(
        collate_eeg_text,
        max_samples=max_samples,
        time_fit_mode=time_fit_mode,
    )

    if train_batch_mode == "multi_positive":
        target_ids_for_sampler = [record.target_id for record in dataset.records]
        train_loader = DataLoader(
            dataset,
            batch_sampler=MultiPositiveTargetBatchSampler(
                indices=train_idx,
                target_ids=target_ids_for_sampler,
                targets_per_batch=targets_per_batch,
                views_per_target=views_per_target,
                shuffle=True,
                seed=seed,
                view_groups_per_target_per_epoch=(
                    view_groups_per_target_per_epoch
                ),
                drop_last=drop_last_target_batch,
            ),
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
    else:
        train_loader = make_loader(
            dataset,
            train_idx,
            batch_size,
            collate_fn,
            num_workers,
            shuffle=True,
            unique_target_per_batch=unique_target_per_batch,
            seed=seed,
            min_batch_size=min_train_batch_size,
        )
    val_loader = make_loader(
        dataset,
        val_idx,
        batch_size,
        collate_fn,
        num_workers,
        shuffle=False,
        unique_target_per_batch=unique_target_per_batch,
        seed=seed + 1,
        min_batch_size=1,
    )
    test_loader = (
        make_loader(
            dataset,
            test_idx,
            batch_size,
            collate_fn,
            num_workers,
            shuffle=False,
            unique_target_per_batch=unique_target_per_batch,
            seed=seed + 2,
            min_batch_size=1,
        )
        if test_idx
        else None
    )

    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
    split_group_ids = {
        split: sorted({dataset.records[index].split_group_id for index in indices})
        for split, indices in split_indices.items()
    }
    target_ids = {
        split: {dataset.records[index].target_id for index in indices}
        for split, indices in split_indices.items()
    }
    if set(split_group_ids["train"]) & set(split_group_ids["val"]):
        raise RuntimeError("split_group_id leakage between train and val")
    if set(split_group_ids["train"]) & set(split_group_ids["test"]):
        raise RuntimeError("split_group_id leakage between train and test")
    if set(split_group_ids["val"]) & set(split_group_ids["test"]):
        raise RuntimeError("split_group_id leakage between val and test")

    train_rows_used = len(train_idx)
    train_rows_dropped = 0
    if isinstance(train_loader.batch_sampler, UniqueTargetBatchSampler):
        train_rows_used = train_loader.batch_sampler.retained_samples
        train_rows_dropped = train_loader.batch_sampler.dropped_samples
    multi_positive_sampler = (
        train_loader.batch_sampler
        if isinstance(train_loader.batch_sampler, MultiPositiveTargetBatchSampler)
        else None
    )
    if multi_positive_sampler is not None:
        train_rows_used = multi_positive_sampler.samples_per_epoch
        train_rows_dropped = (
            len(train_idx) - multi_positive_sampler.unique_samples_first_epoch
        )

    print(
        f"split_rows train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"split_groups train={len(split_group_ids['train'])} val={len(split_group_ids['val'])} "
        f"test={len(split_group_ids['test'])} canonical_targets={len(set().union(*target_ids.values()))} "
        f"subjects={len(dataset.subject_to_id)} model={model_name} "
        f"subject_layers={subject_layers_enabled} "
        f"time_fit_mode={time_fit_mode} max_samples={max_samples} "
        f"train_batch_mode={train_batch_mode} "
        f"unique_target_per_batch={unique_target_per_batch} "
        f"min_train_batch_size={min_train_batch_size} "
        f"train_rows_used={train_rows_used} train_rows_dropped={train_rows_dropped}"
    )

    protocol = {
        "identity": "exact float32 embedding SHA256",
        "splitter": "SHA256(group_id) + seeded Python Random",
        "split_group": "canonical target UID unless supplied by manifest",
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "split_group_ids": split_group_ids,
        "temporal_window": {
            "fit_mode": time_fit_mode,
            "max_samples": max_samples,
        },
        "batching": {
            "batch_size": batch_size,
            "train_batch_mode": train_batch_mode,
            "unique_target_per_batch": unique_target_per_batch,
            "min_train_batch_size": min_train_batch_size,
            "train_rows_used": train_rows_used,
            "train_rows_dropped": train_rows_dropped,
            "shuffle_batches_after_construction": True,
            "multi_positive": (
                {
                    "targets_per_batch": targets_per_batch,
                    "views_per_target": views_per_target,
                    "view_groups_per_target_per_epoch": (
                        view_groups_per_target_per_epoch
                    ),
                    "drop_last_target_batch": drop_last_target_batch,
                    "samples_per_epoch": multi_positive_sampler.samples_per_epoch,
                    "unique_samples_first_epoch": (
                        multi_positive_sampler.unique_samples_first_epoch
                    ),
                    "target_groups_per_epoch": (
                        multi_positive_sampler.target_groups_per_epoch
                    ),
                }
                if multi_positive_sampler is not None
                else None
            ),
        },
        "subject_conditioning": {
            "enabled": subject_layers_enabled,
            "layer": (
                "per-subject linear after shared initial projection"
                if model_name.lower().replace("-", "_")
                in {"simpleconv_timeagg", "simpleconv_time_agg"}
                else "per-subject linear at encoder input"
            ),
            "subject_to_id": dataset.subject_to_id,
        },
    }

    device = torch.device(device_name)
    model = build_eeg_encoder(model_name, **model_kwargs).to(device)
    effective_model_config = {"name": model_name, **model_kwargs}
    print(f"model_parameters={sum(parameter.numel() for parameter in model.parameters())}")
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
        current_metrics = {f"val_{name}": value for name, value in val_metrics.items()}
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
                    "model_config": effective_model_config,
                    "subject_to_id": dataset.subject_to_id,
                    "val_metrics": current_metrics,
                    "protocol": protocol,
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
            f"val_full_macro_top10={val_metrics['full_macro_top10']:.4f} "
            f"val_full_median_rank={val_metrics['full_median_rank']:.2f} "
            f"val_full_instance_top10={val_metrics['full_instance_top10']:.4f} "
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
            f"test_full_top1={test_metrics['full_top1']:.4f} test_full_top10={test_metrics['full_top10']:.4f} "
            f"test_full_macro_top10={test_metrics['full_macro_top10']:.4f} "
            f"test_full_median_rank={test_metrics['full_median_rank']:.2f} "
            f"test_full_instance_top10={test_metrics['full_instance_top10']:.4f}"
        )


if __name__ == "__main__":
    main()
