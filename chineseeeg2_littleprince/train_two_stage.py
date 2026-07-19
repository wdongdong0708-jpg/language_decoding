from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from chineseeeg2_littleprince.data import (
    EEGTextDataset,
    collate_eeg_text,
    split_indices_by_group,
)
from chineseeeg2_littleprince.data.coarse_clusters import (
    CoarseEEGTextDataset,
    collate_coarse_eeg_text,
)
from chineseeeg2_littleprince.models import build_eeg_encoder
from chineseeeg2_littleprince.train import (
    METRIC_MODES,
    UniqueTargetBatchSampler,
    checkpoint_is_better,
    evaluate,
    load_config,
    make_loader,
    resolve_manifest_path,
    run_epoch,
    text_embedding_for_similarity,
)


TRANSFER_EXCLUDED_PREFIXES = (
    "head.",
    "text_projection.",
)


def _absolute_output_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def extract_eeg_backbone_state_dict(model) -> dict[str, torch.Tensor]:
    """Keep only EEG representation parameters shared by coarse and fine stages."""

    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith(TRANSFER_EXCLUDED_PREFIXES)
    }


def load_eeg_backbone_state_dict(model, state_dict: dict[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_backbone = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(TRANSFER_EXCLUDED_PREFIXES)
    ]
    if missing_backbone or incompatible.unexpected_keys:
        raise ValueError(
            "Incompatible coarse backbone checkpoint: "
            f"missing_backbone={missing_backbone} "
            f"unexpected={incompatible.unexpected_keys}"
        )


def _split_and_loaders(
    dataset: EEGTextDataset,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    batch_size: int,
    max_samples: int,
    num_workers: int,
    unique_target_per_batch: bool,
    min_train_batch_size: int,
    collate_function=collate_eeg_text,
):
    train_idx, val_idx, test_idx = split_indices_by_group(
        dataset.records, val_fraction, test_fraction, seed
    )
    if not train_idx or not val_idx or not test_idx:
        raise ValueError(
            "Two-stage training requires non-empty train/val/test splits: "
            f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}"
        )
    collate_fn = partial(collate_function, max_samples=max_samples)
    common = {
        "dataset": dataset,
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "unique_target_per_batch": unique_target_per_batch,
    }
    train_loader = make_loader(
        indices=train_idx,
        shuffle=True,
        seed=seed,
        min_batch_size=min_train_batch_size,
        **common,
    )
    val_loader = make_loader(
        indices=val_idx,
        shuffle=False,
        seed=seed + 1,
        min_batch_size=1,
        **common,
    )
    test_loader = make_loader(
        indices=test_idx,
        shuffle=False,
        seed=seed + 2,
        min_batch_size=1,
        **common,
    )
    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
    split_groups = {
        split: {dataset.records[index].split_group_id for index in indices}
        for split, indices in split_indices.items()
    }
    if any(
        split_groups[left] & split_groups[right]
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]
    ):
        raise RuntimeError("split_group_id leakage in two-stage protocol")
    return train_loader, val_loader, test_loader, split_indices, split_groups


def _macro_accuracy(
    correct: torch.Tensor,
    target_id: torch.Tensor,
) -> float:
    values = [
        correct[target_id == value].to(torch.float32).mean()
        for value in torch.unique(target_id)
    ]
    return float(torch.stack(values).mean())


def _coarse_candidate_embeddings(model, targets, device: torch.device) -> torch.Tensor:
    frozen = torch.from_numpy(targets.coarse_embeddings).to(device)
    return text_embedding_for_similarity(model, frozen)


def run_coarse_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    targets,
    temperature: float,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "cos": 0.0, "top1": 0.0, "top5": 0.0}
    total_rows = 0
    for batch in loader:
        eeg = batch["eeg"].to(device)
        mask = batch["mask"].to(device)
        subject_id = batch["subject_id"].to(device)
        coarse_id = batch["coarse_id"].to(device)
        predictions = model(eeg, mask, subject_id=subject_id)
        candidates = _coarse_candidate_embeddings(model, targets, device)
        logits = F.normalize(predictions, dim=-1) @ F.normalize(candidates, dim=-1).T
        logits = logits / temperature
        loss = F.cross_entropy(logits, coarse_id)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = eeg.shape[0]
        total_rows += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        totals["cos"] += float(
            F.cosine_similarity(
                predictions.detach(), candidates.detach().index_select(0, coarse_id)
            ).mean()
        ) * batch_size
        totals["top1"] += float(
            logits.detach().argmax(dim=1).eq(coarse_id).to(torch.float32).mean()
        ) * batch_size
        totals["top5"] += float(
            logits.detach().topk(min(5, logits.shape[1]), dim=1).indices
            .eq(coarse_id[:, None])
            .any(dim=1)
            .to(torch.float32)
            .mean()
        ) * batch_size
    return {name: value / total_rows for name, value in totals.items()}


@torch.no_grad()
def evaluate_coarse(
    model,
    loader,
    device: torch.device,
    targets,
    temperature: float,
) -> dict[str, float]:
    model.eval()
    candidates = _coarse_candidate_embeddings(model, targets, device)
    total_loss = 0.0
    total_rows = 0
    all_logits = []
    all_target = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        mask = batch["mask"].to(device)
        subject_id = batch["subject_id"].to(device)
        coarse_id = batch["coarse_id"].to(device)
        prediction = model(eeg, mask, subject_id=subject_id)
        logits = F.normalize(prediction, dim=-1) @ F.normalize(candidates, dim=-1).T
        logits = logits / temperature
        batch_size = eeg.shape[0]
        total_rows += batch_size
        total_loss += float(F.cross_entropy(logits, coarse_id)) * batch_size
        all_logits.append(logits)
        all_target.append(coarse_id)

    logits = torch.cat(all_logits)
    target = torch.cat(all_target)
    top1_correct = logits.argmax(dim=1).eq(target)
    top5_correct = logits.topk(min(5, logits.shape[1]), dim=1).indices.eq(
        target[:, None]
    ).any(dim=1)
    return {
        "loss": total_loss / total_rows,
        "top1": float(top1_correct.to(torch.float32).mean()),
        "top5": float(top5_correct.to(torch.float32).mean()),
        "macro_top1": _macro_accuracy(top1_correct, target),
        "macro_top5": _macro_accuracy(top5_correct, target),
    }


def _model_kwargs_for_subjects(
    base_kwargs: dict[str, Any],
    dataset: EEGTextDataset,
) -> dict[str, Any]:
    output = dict(base_kwargs)
    if output.get("subject_layers", False):
        required = len(dataset.subject_to_id)
        configured = output.get("n_subjects")
        if configured is not None and int(configured) < required:
            raise ValueError(
                f"model.n_subjects={configured} is smaller than dataset subjects={required}"
            )
        output["n_subjects"] = required if configured is None else int(configured)
    return output


def _stage2_optimizer(
    model,
    *,
    backbone_lr: float,
    task_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_parameters = []
    task_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith(("head.", "text_projection.")):
            task_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": task_parameters, "lr": task_lr},
        ],
        weight_decay=weight_decay,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--coarse-epochs", type=int)
    parser.add_argument("--fine-epochs", type=int)
    parser.add_argument("--coarse-checkpoint", type=Path)
    parser.add_argument("--fine-checkpoint", type=Path)
    parser.add_argument(
        "--fine-init",
        choices=("coarse", "random"),
        default="coarse",
        help="Initialize stage-two EEG backbone from coarse pretraining or keep its seeded random initialization.",
    )
    parser.add_argument("--device")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    manifest_path = resolve_manifest_path(config["manifest"], config_path)
    cluster_config = config.get("coarse_clusters") or {}
    coarse_cluster_path = resolve_manifest_path(
        cluster_config["path"], config_path
    )
    coarse_config = dict(config.get("coarse_pretrain") or {})
    fine_config = dict(config.get("fine_tune") or {})
    if not coarse_config or not fine_config:
        raise ValueError("Two-stage config needs coarse_pretrain and fine_tune mappings")

    seed = int(config.get("seed", 42))
    batch_size = int(config.get("batch_size", 128))
    max_samples = int(config.get("max_samples", 1300))
    num_workers = int(config.get("num_workers", 0))
    normalize_eeg = bool(config.get("normalize_eeg", True))
    subjects = config.get("subjects")
    val_fraction = float(config.get("val_fraction", 0.1))
    test_fraction = float(config.get("test_fraction", 0.1))
    unique_target_per_batch = bool(config.get("unique_target_per_batch", True))
    min_train_batch_size = int(config.get("min_train_batch_size", 32))
    device = torch.device(
        args.device
        or config.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_config = dict(config.get("model") or {})
    model_name = str(model_config.pop("name", "simpleconv_timeagg"))
    if model_name.lower().replace("-", "_") not in {
        "simpleconv_timeagg",
        "simpleconv_time_agg",
    }:
        raise ValueError("Two-stage training currently requires simpleconv_timeagg")

    coarse_epochs = int(args.coarse_epochs or coarse_config.get("epochs", 10))
    fine_epochs = int(args.fine_epochs or fine_config.get("epochs", 40))
    if coarse_epochs <= 0 or fine_epochs <= 0:
        raise ValueError("Both two-stage epoch counts must be positive")

    coarse_dataset = CoarseEEGTextDataset(
        manifest_path,
        normalize_eeg=normalize_eeg,
        subjects=subjects,
        coarse_cluster_path=coarse_cluster_path,
    )
    coarse_loaders = _split_and_loaders(
        coarse_dataset,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        batch_size=batch_size,
        max_samples=max_samples,
        num_workers=num_workers,
        unique_target_per_batch=unique_target_per_batch,
        min_train_batch_size=min_train_batch_size,
        collate_function=collate_coarse_eeg_text,
    )
    coarse_train_loader, coarse_val_loader, _, coarse_indices, coarse_groups = coarse_loaders
    coarse_model_kwargs = _model_kwargs_for_subjects(model_config, coarse_dataset)

    torch.manual_seed(seed)
    coarse_model = build_eeg_encoder(model_name, **coarse_model_kwargs).to(device)
    coarse_lr = float(coarse_config.get("learning_rate", 1e-4))
    coarse_weight_decay = float(coarse_config.get("weight_decay", 1e-4))
    coarse_temperature = float(coarse_config.get("temperature", 0.1))
    coarse_optimizer = torch.optim.AdamW(
        coarse_model.parameters(), lr=coarse_lr, weight_decay=coarse_weight_decay
    )
    coarse_checkpoint_path = _absolute_output_path(
        args.coarse_checkpoint
        or coarse_config.get(
            "checkpoint_path", "checkpoints/two_stage_coarse_pretrain_best.pt"
        )
    )
    coarse_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    coarse_patience = int(coarse_config.get("early_stopping_patience", 5))
    coarse_metric_name = str(coarse_config.get("checkpoint_metric", "macro_top5"))
    if coarse_metric_name not in {"top1", "top5", "macro_top1", "macro_top5"}:
        raise ValueError("Unsupported coarse checkpoint metric")

    retained = (
        coarse_train_loader.batch_sampler.retained_samples
        if isinstance(coarse_train_loader.batch_sampler, UniqueTargetBatchSampler)
        else len(coarse_indices["train"])
    )
    print(
        "stage=coarse "
        f"split_rows train={len(coarse_indices['train'])} val={len(coarse_indices['val'])} "
        f"test={len(coarse_indices['test'])} "
        f"split_groups train={len(coarse_groups['train'])} val={len(coarse_groups['val'])} "
        f"test={len(coarse_groups['test'])} "
        f"fine_targets={coarse_dataset.coarse_targets.active_fine_count} "
        f"coarse_targets={coarse_dataset.coarse_targets.coarse_count} "
        f"train_rows_used={retained} "
        f"model_parameters={sum(parameter.numel() for parameter in coarse_model.parameters())}"
    )

    best_coarse_metric = None
    best_coarse_epoch = 0
    coarse_without_improvement = 0
    for epoch in range(1, coarse_epochs + 1):
        train_metrics = run_coarse_epoch(
            coarse_model,
            coarse_train_loader,
            coarse_optimizer,
            device,
            coarse_dataset.coarse_targets,
            coarse_temperature,
        )
        val_metrics = evaluate_coarse(
            coarse_model,
            coarse_val_loader,
            device,
            coarse_dataset.coarse_targets,
            coarse_temperature,
        )
        metric = val_metrics[coarse_metric_name]
        improved = checkpoint_is_better(metric, best_coarse_metric, "max")
        if improved:
            best_coarse_metric = metric
            best_coarse_epoch = epoch
            coarse_without_improvement = 0
            torch.save(
                {
                    "stage": "coarse_pretrain",
                    "epoch": epoch,
                    "checkpoint_metric": coarse_metric_name,
                    "checkpoint_metric_value": metric,
                    "model_state_dict": coarse_model.state_dict(),
                    "backbone_state_dict": extract_eeg_backbone_state_dict(coarse_model),
                    "model_config": {"name": model_name, **coarse_model_kwargs},
                    "subject_to_id": coarse_dataset.subject_to_id,
                    "val_metrics": val_metrics,
                    "protocol": {
                        "loss": "coarse-only InfoNCE against all 32 prototypes",
                        "fine_targets_used_for_split": (
                            coarse_dataset.coarse_targets.active_fine_count
                        ),
                        "coarse_targets": coarse_dataset.coarse_targets.coarse_count,
                        "coarse_cluster_path": str(coarse_cluster_path),
                    },
                },
                coarse_checkpoint_path,
            )
        else:
            coarse_without_improvement += 1
        print(
            f"stage=coarse epoch={epoch:03d} train_loss={train_metrics['loss']:.5f} "
            f"train_top1={train_metrics['top1']:.4f} train_top5={train_metrics['top5']:.4f} "
            f"val_loss={val_metrics['loss']:.5f} val_top1={val_metrics['top1']:.4f} "
            f"val_top5={val_metrics['top5']:.4f} val_macro_top1={val_metrics['macro_top1']:.4f} "
            f"val_macro_top5={val_metrics['macro_top5']:.4f} "
            f"best_{coarse_metric_name}={best_coarse_metric:.4f} best_epoch={best_coarse_epoch}"
        )
        if coarse_patience > 0 and coarse_without_improvement >= coarse_patience:
            break

    coarse_checkpoint = torch.load(
        coarse_checkpoint_path, map_location="cpu", weights_only=False
    )
    print(
        f"stage=coarse loaded_best path={coarse_checkpoint_path} "
        f"epoch={coarse_checkpoint['epoch']} "
        f"{coarse_metric_name}={coarse_checkpoint['checkpoint_metric_value']:.4f}"
    )

    # Stage two restores the exact original ROW task, including chapter-number targets.
    fine_dataset = EEGTextDataset(
        manifest_path,
        normalize_eeg=normalize_eeg,
        subjects=subjects,
    )
    if fine_dataset.subject_to_id != coarse_dataset.subject_to_id:
        raise ValueError("Subject identity mapping changed between coarse and fine stages")
    fine_loaders = _split_and_loaders(
        fine_dataset,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        batch_size=batch_size,
        max_samples=max_samples,
        num_workers=num_workers,
        unique_target_per_batch=unique_target_per_batch,
        min_train_batch_size=min_train_batch_size,
    )
    fine_train_loader, fine_val_loader, fine_test_loader, fine_indices, fine_groups = fine_loaders
    fine_model_kwargs = _model_kwargs_for_subjects(model_config, fine_dataset)

    torch.manual_seed(seed + 1_000)
    fine_model = build_eeg_encoder(model_name, **fine_model_kwargs).to(device)
    if args.fine_init == "coarse":
        load_eeg_backbone_state_dict(
            fine_model, coarse_checkpoint["backbone_state_dict"]
        )
    backbone_lr = float(fine_config.get("backbone_learning_rate", 3e-5))
    task_lr = float(fine_config.get("task_learning_rate", 1e-4))
    fine_weight_decay = float(fine_config.get("weight_decay", 1e-4))
    fine_temperature = float(fine_config.get("temperature", 0.05))
    fine_optimizer = _stage2_optimizer(
        fine_model,
        backbone_lr=backbone_lr,
        task_lr=task_lr,
        weight_decay=fine_weight_decay,
    )
    fine_checkpoint_path = _absolute_output_path(
        args.fine_checkpoint
        or fine_config.get("checkpoint_path", "checkpoints/two_stage_fine_best.pt")
    )
    fine_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    fine_metric_name = str(fine_config.get("checkpoint_metric", "val_full_macro_top10"))
    if fine_metric_name not in METRIC_MODES:
        raise ValueError(f"Unsupported fine checkpoint metric {fine_metric_name!r}")
    fine_patience = int(fine_config.get("early_stopping_patience", 8))

    fine_retained = (
        fine_train_loader.batch_sampler.retained_samples
        if isinstance(fine_train_loader.batch_sampler, UniqueTargetBatchSampler)
        else len(fine_indices["train"])
    )
    print(
        "stage=fine "
        f"split_rows train={len(fine_indices['train'])} val={len(fine_indices['val'])} "
        f"test={len(fine_indices['test'])} "
        f"split_groups train={len(fine_groups['train'])} val={len(fine_groups['val'])} "
        f"test={len(fine_groups['test'])} canonical_targets="
        f"{len(set().union(*fine_groups.values()))} train_rows_used={fine_retained} "
        f"fine_init={args.fine_init} "
        f"backbone_lr={backbone_lr} task_lr={task_lr} "
        f"model_parameters={sum(parameter.numel() for parameter in fine_model.parameters())}"
    )

    fine_protocol = {
        "initialization": (
            "EEG backbone from coarse-only pretraining"
            if args.fine_init == "coarse"
            else f"seeded random initialization (seed={seed + 1_000})"
        ),
        "fine_loss": "original fine-only in-batch contrastive loss",
        "backbone_learning_rate": backbone_lr,
        "task_learning_rate": task_lr,
        "control_variable": "EEG backbone initialization only",
    }
    if args.fine_init == "coarse":
        fine_protocol.update(
            {
                "coarse_checkpoint": str(coarse_checkpoint_path),
                "coarse_checkpoint_epoch": coarse_checkpoint["epoch"],
                "coarse_checkpoint_metric_value": coarse_checkpoint[
                    "checkpoint_metric_value"
                ],
            }
        )

    best_fine_metric = None
    best_fine_epoch = 0
    fine_without_improvement = 0
    fine_checkpoint_mode = METRIC_MODES[fine_metric_name]
    for epoch in range(1, fine_epochs + 1):
        train_loss, train_cos, train_top1, train_top10 = run_epoch(
            fine_model,
            fine_train_loader,
            fine_optimizer,
            device,
            fine_temperature,
        )
        val_metrics = evaluate(
            fine_model, fine_val_loader, device, fine_temperature
        )
        current_metrics = {f"val_{name}": value for name, value in val_metrics.items()}
        metric = current_metrics[fine_metric_name]
        improved = checkpoint_is_better(
            metric, best_fine_metric, fine_checkpoint_mode
        )
        if improved:
            best_fine_metric = metric
            best_fine_epoch = epoch
            fine_without_improvement = 0
            torch.save(
                {
                    "stage": "fine_tune",
                    "epoch": epoch,
                    "checkpoint_metric": fine_metric_name,
                    "checkpoint_metric_value": metric,
                    "model_state_dict": fine_model.state_dict(),
                    "optimizer_state_dict": fine_optimizer.state_dict(),
                    "model_config": {"name": model_name, **fine_model_kwargs},
                    "subject_to_id": fine_dataset.subject_to_id,
                    "val_metrics": current_metrics,
                    "protocol": fine_protocol,
                },
                fine_checkpoint_path,
            )
        else:
            fine_without_improvement += 1
        print(
            f"stage=fine epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"train_cos={train_cos:.4f} train_top1={train_top1:.4f} "
            f"train_top10={train_top10:.4f} val_loss={val_metrics['loss']:.5f} "
            f"val_full_top1={val_metrics['full_top1']:.4f} "
            f"val_full_top10={val_metrics['full_top10']:.4f} "
            f"val_full_macro_top10={val_metrics['full_macro_top10']:.4f} "
            f"val_full_median_rank={val_metrics['full_median_rank']:.2f} "
            f"best_{fine_metric_name}={best_fine_metric:.4f} best_epoch={best_fine_epoch}"
        )
        if fine_patience > 0 and fine_without_improvement >= fine_patience:
            break

    fine_checkpoint = torch.load(
        fine_checkpoint_path, map_location=device, weights_only=False
    )
    fine_model.load_state_dict(fine_checkpoint["model_state_dict"])
    print(
        f"stage=fine loaded_best path={fine_checkpoint_path} "
        f"epoch={fine_checkpoint['epoch']} "
        f"{fine_metric_name}={fine_checkpoint['checkpoint_metric_value']:.4f}"
    )
    test_metrics = evaluate(
        fine_model, fine_test_loader, device, fine_temperature
    )
    print(
        f"test_loss={test_metrics['loss']:.5f} test_cos={test_metrics['cos']:.4f} "
        f"test_top1={test_metrics['top1']:.4f} test_top10={test_metrics['top10']:.4f} "
        f"test_full_top1={test_metrics['full_top1']:.4f} "
        f"test_full_top10={test_metrics['full_top10']:.4f} "
        f"test_full_macro_top10={test_metrics['full_macro_top10']:.4f} "
        f"test_full_median_rank={test_metrics['full_median_rank']:.2f} "
        f"test_full_instance_top10={test_metrics['full_instance_top10']:.4f}"
    )


if __name__ == "__main__":
    main()
