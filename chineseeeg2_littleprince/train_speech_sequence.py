from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from chineseeeg2_littleprince.data import EEGSpeechSequenceDataset, collate_eeg_speech_sequence
from chineseeeg2_littleprince.models import TemporalConvEEGSequenceEncoder
from chineseeeg2_littleprince.train import (
    DEFAULTS,
    METRIC_MODES,
    UniqueTextBatchSampler,
    checkpoint_is_better,
    coalesce,
    load_config,
    make_loader,
    nested_get,
    resolve_manifest_path,
    split_indices_by_text,
)


def sequence_similarity_logits(
    eeg_sequence: torch.Tensor,
    speech_sequence: torch.Tensor,
    eeg_mask: torch.Tensor,
    speech_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    eeg_sequence = F.normalize(eeg_sequence, dim=-1)
    speech_sequence = F.normalize(speech_sequence, dim=-1)
    frame_similarity = torch.einsum("bld,nld->bnl", eeg_sequence, speech_sequence)
    valid = eeg_mask.unsqueeze(1) & speech_mask.unsqueeze(0)
    scores = (frame_similarity * valid.to(frame_similarity.dtype)).sum(dim=-1)
    scores = scores / valid.sum(dim=-1).clamp_min(1).to(frame_similarity.dtype)
    return scores / temperature


def sequence_cosine_mean(
    eeg_sequence: torch.Tensor,
    speech_sequence: torch.Tensor,
    eeg_mask: torch.Tensor,
    speech_mask: torch.Tensor,
) -> torch.Tensor:
    eeg_sequence = F.normalize(eeg_sequence, dim=-1)
    speech_sequence = F.normalize(speech_sequence, dim=-1)
    frame_similarity = (eeg_sequence * speech_sequence).sum(dim=-1)
    valid = eeg_mask & speech_mask
    return (frame_similarity * valid.to(frame_similarity.dtype)).sum() / valid.sum().clamp_min(1)


def eeg_to_speech_sequence_loss(
    eeg_sequence: torch.Tensor,
    speech_sequence: torch.Tensor,
    eeg_mask: torch.Tensor,
    speech_mask: torch.Tensor,
    label_id: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = sequence_similarity_logits(eeg_sequence, speech_sequence, eeg_mask, speech_mask, temperature)
    positive_mask = label_id.unsqueeze(1).eq(label_id.unsqueeze(0))
    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = -(log_probs * positive_mask.to(log_probs.dtype)).sum(dim=1) / positive_counts
    return loss.mean(), logits


def sequence_retrieval_topk(logits: torch.Tensor, label_id: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.shape[1])
    predictions = logits.topk(k, dim=1).indices
    positive_mask = label_id.unsqueeze(1).eq(label_id.unsqueeze(0))
    return positive_mask.gather(dim=1, index=predictions).any(dim=1).float().mean()


def _unique_sequence_candidates(
    speech_sequence: torch.Tensor,
    speech_mask: torch.Tensor,
    label_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first_by_text_id: dict[int, int] = {}
    for index, text_id in enumerate(label_id.detach().cpu().tolist()):
        first_by_text_id.setdefault(int(text_id), index)

    ordered_text_ids = sorted(first_by_text_id)
    candidate_indices = torch.tensor(
        [first_by_text_id[text_id] for text_id in ordered_text_ids],
        dtype=torch.long,
        device=speech_sequence.device,
    )
    return (
        speech_sequence.index_select(0, candidate_indices),
        speech_mask.index_select(0, candidate_indices),
        torch.tensor(ordered_text_ids, dtype=label_id.dtype, device=label_id.device),
    )


def full_sequence_retrieval_topk(
    eeg_sequence: torch.Tensor,
    speech_sequence: torch.Tensor,
    eeg_mask: torch.Tensor,
    speech_mask: torch.Tensor,
    label_id: torch.Tensor,
    k: int,
    chunk_size: int = 256,
) -> torch.Tensor:
    speech_candidates, candidate_masks, candidate_text_ids = _unique_sequence_candidates(
        speech_sequence,
        speech_mask,
        label_id,
    )
    k = min(k, speech_candidates.shape[0])

    total_correct = 0.0
    total = 0
    for start in range(0, eeg_sequence.shape[0], chunk_size):
        stop = min(start + chunk_size, eeg_sequence.shape[0])
        logits = sequence_similarity_logits(
            eeg_sequence[start:stop],
            speech_candidates,
            eeg_mask[start:stop],
            candidate_masks,
            temperature=1.0,
        )
        predictions = logits.topk(k, dim=1).indices
        query_text_ids = label_id[start:stop]
        positive_mask = query_text_ids.unsqueeze(1).eq(candidate_text_ids.unsqueeze(0))
        total_correct += float(positive_mask.gather(dim=1, index=predictions).any(dim=1).float().sum())
        total += stop - start

    return torch.tensor(total_correct / max(total, 1), device=eeg_sequence.device)


def run_epoch(model, loader, optimizer, device: torch.device, temperature: float) -> tuple[float, float, float, float]:
    model.train()
    total_loss = 0.0
    total_cos = 0.0
    total_top1 = 0.0
    total_top10 = 0.0
    total = 0
    for batch in loader:
        eeg = batch["eeg"].to(device)
        eeg_mask = batch["eeg_mask"].to(device)
        speech = batch["speech"].to(device)
        speech_mask = batch["speech_mask"].to(device)
        label_id = batch["label_id"].to(device)

        pred, pred_mask = model(eeg, eeg_mask, return_mask=True)
        loss, logits = eeg_to_speech_sequence_loss(pred, speech, pred_mask, speech_mask, label_id, temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = eeg.shape[0]
        total += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_cos += float(sequence_cosine_mean(pred.detach(), speech, pred_mask, speech_mask)) * batch_size
        total_top1 += float(sequence_retrieval_topk(logits.detach(), label_id, k=1)) * batch_size
        total_top10 += float(sequence_retrieval_topk(logits.detach(), label_id, k=10)) * batch_size
    return total_loss / total, total_cos / total, total_top1 / total, total_top10 / total


@torch.no_grad()
def evaluate(model, loader, device: torch.device, temperature: float) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_cos = 0.0
    total_top1 = 0.0
    total_top10 = 0.0
    total = 0
    all_pred = []
    all_pred_mask = []
    all_speech = []
    all_speech_mask = []
    all_label_id = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        eeg_mask = batch["eeg_mask"].to(device)
        speech = batch["speech"].to(device)
        speech_mask = batch["speech_mask"].to(device)
        label_id = batch["label_id"].to(device)
        pred, pred_mask = model(eeg, eeg_mask, return_mask=True)
        loss, logits = eeg_to_speech_sequence_loss(pred, speech, pred_mask, speech_mask, label_id, temperature)
        batch_size = eeg.shape[0]
        total += batch_size
        total_loss += float(loss) * batch_size
        total_cos += float(sequence_cosine_mean(pred, speech, pred_mask, speech_mask)) * batch_size
        total_top1 += float(sequence_retrieval_topk(logits, label_id, k=1)) * batch_size
        total_top10 += float(sequence_retrieval_topk(logits, label_id, k=10)) * batch_size
        all_pred.append(pred.detach().cpu())
        all_pred_mask.append(pred_mask.detach().cpu())
        all_speech.append(speech.detach().cpu())
        all_speech_mask.append(speech_mask.detach().cpu())
        all_label_id.append(label_id.detach().cpu())

    pred_all = torch.cat(all_pred, dim=0)
    pred_mask_all = torch.cat(all_pred_mask, dim=0)
    speech_all = torch.cat(all_speech, dim=0)
    speech_mask_all = torch.cat(all_speech_mask, dim=0)
    label_id_all = torch.cat(all_label_id, dim=0)
    return {
        "loss": total_loss / total,
        "cos": total_cos / total,
        "top1": total_top1 / total,
        "top10": total_top10 / total,
        "full_top1": float(
            full_sequence_retrieval_topk(pred_all, speech_all, pred_mask_all, speech_mask_all, label_id_all, k=1)
        ),
        "full_top10": float(
            full_sequence_retrieval_topk(pred_all, speech_all, pred_mask_all, speech_mask_all, label_id_all, k=10)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sequence-frames", type=int, default=None)
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

    model_config = dict(config.get("model", {}))
    seed = int(coalesce(args.seed, config.get("seed"), DEFAULTS["seed"]))
    batch_size = int(coalesce(args.batch_size, config.get("batch_size"), DEFAULTS["batch_size"]))
    max_samples = int(coalesce(args.max_samples, config.get("max_samples"), DEFAULTS["max_samples"]))
    sequence_frames = int(coalesce(args.sequence_frames, config.get("sequence_frames"), model_config.get("sequence_frames"), 64))
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
            "checkpoints/littleprince_sentence_audio_sequence_best.pt",
        )
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    device_name = str(coalesce(args.device, config.get("device"), "cuda" if torch.cuda.is_available() else "cpu"))

    model_kwargs: dict[str, Any] = {
        "in_channels": 128,
        "hidden_channels": 128,
        "embedding_dim": 1024,
        "sequence_frames": sequence_frames,
        "dropout": 0.2,
    }
    model_kwargs.update(model_config)
    model_kwargs["sequence_frames"] = sequence_frames

    torch.manual_seed(seed)
    manifest_path = resolve_manifest_path(manifest, config_path)
    dataset = EEGSpeechSequenceDataset(manifest_path, normalize_eeg=normalize_eeg)
    train_idx, val_idx, test_idx = split_indices_by_text(dataset.records, val_fraction, test_fraction, seed)
    collate_fn = partial(collate_eeg_speech_sequence, max_samples=max_samples, sequence_frames=sequence_frames)

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
        f"sequence_frames={sequence_frames} unique_text_per_batch={unique_text_per_batch}"
    )

    device = torch.device(device_name)
    model = TemporalConvEEGSequenceEncoder(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_mode = METRIC_MODES[checkpoint_metric]
    best_metric = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_cos, train_top1, train_top10 = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            contrastive_temperature,
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
                    "target": "sentence_audio_sequence",
                    "sequence_frames": sequence_frames,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} train_cos={train_cos:.4f} "
            f"train_top1={train_top1:.4f} train_top10={train_top10:.4f} "
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
