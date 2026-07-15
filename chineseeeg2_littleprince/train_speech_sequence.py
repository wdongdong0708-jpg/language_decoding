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


SEQUENCE_METRIC_MODES = {
    **METRIC_MODES,
    "val_speaker_full_top1": "max",
    "val_speaker_full_top10": "max",
    "val_dual_full_top1": "max",
    "val_dual_full_top10": "max",
}


def sequence_similarity_logits(
    eeg_sequence: torch.Tensor,
    speech_sequence: torch.Tensor,
    eeg_mask: torch.Tensor,
    speech_mask: torch.Tensor,
    temperature: float,
    alignment_max_lag: int = 0,
    alignment_min_frames: int = 1,
) -> torch.Tensor:
    """Lag-tolerant frame alignment over padded, variable-length sequences."""
    if alignment_max_lag < 0:
        raise ValueError(f"alignment_max_lag must be non-negative, got {alignment_max_lag}")
    if alignment_min_frames <= 0:
        raise ValueError(f"alignment_min_frames must be positive, got {alignment_min_frames}")
    eeg_sequence = F.normalize(eeg_sequence, dim=-1)
    speech_sequence = F.normalize(speech_sequence, dim=-1)
    eeg_frames = eeg_sequence.shape[1]
    speech_frames = speech_sequence.shape[1]
    scores_by_lag = []
    for lag in range(-alignment_max_lag, alignment_max_lag + 1):
        if lag >= 0:
            frame_count = min(eeg_frames, speech_frames - lag)
            eeg_start, speech_start = 0, lag
        else:
            frame_count = min(eeg_frames + lag, speech_frames)
            eeg_start, speech_start = -lag, 0
        if frame_count <= 0:
            continue

        eeg_part = eeg_sequence[:, eeg_start : eeg_start + frame_count]
        speech_part = speech_sequence[:, speech_start : speech_start + frame_count]
        frame_similarity = torch.einsum("btd,ntd->bnt", eeg_part, speech_part)
        valid = (
            eeg_mask[:, eeg_start : eeg_start + frame_count].unsqueeze(1)
            & speech_mask[:, speech_start : speech_start + frame_count].unsqueeze(0)
        )
        valid_count = valid.sum(dim=-1)
        scores = (frame_similarity * valid.to(frame_similarity.dtype)).sum(dim=-1)
        scores = scores / valid_count.clamp_min(1).to(frame_similarity.dtype)
        scores_by_lag.append(scores.masked_fill(valid_count < alignment_min_frames, float("-inf")))

    if not scores_by_lag:
        raise ValueError("No overlapping frames are available for sequence alignment")
    return torch.stack(scores_by_lag, dim=-1).max(dim=-1).values / temperature


def sequence_cosine_mean(
    eeg_sequence: torch.Tensor,
    speech_sequence: torch.Tensor,
    eeg_mask: torch.Tensor,
    speech_mask: torch.Tensor,
    alignment_max_lag: int = 0,
    alignment_min_frames: int = 1,
) -> torch.Tensor:
    scores = sequence_similarity_logits(
        eeg_sequence,
        speech_sequence,
        eeg_mask,
        speech_mask,
        temperature=1.0,
        alignment_max_lag=alignment_max_lag,
        alignment_min_frames=alignment_min_frames,
    )
    return scores.diagonal().mean()


def eeg_to_speech_sequence_loss(
    eeg_sequence: torch.Tensor,
    speech_sequence: torch.Tensor,
    eeg_mask: torch.Tensor,
    speech_mask: torch.Tensor,
    label_id: torch.Tensor,
    temperature: float,
    alignment_max_lag: int = 0,
    alignment_min_frames: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = sequence_similarity_logits(
        eeg_sequence,
        speech_sequence,
        eeg_mask,
        speech_mask,
        temperature,
        alignment_max_lag=alignment_max_lag,
        alignment_min_frames=alignment_min_frames,
    )
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
    alignment_max_lag: int = 0,
    alignment_min_frames: int = 1,
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
            alignment_max_lag=alignment_max_lag,
            alignment_min_frames=alignment_min_frames,
        )
        predictions = logits.topk(k, dim=1).indices
        query_text_ids = label_id[start:stop]
        positive_mask = query_text_ids.unsqueeze(1).eq(candidate_text_ids.unsqueeze(0))
        total_correct += float(positive_mask.gather(dim=1, index=predictions).any(dim=1).float().sum())
        total += stop - start

    return torch.tensor(total_correct / max(total, 1), device=eeg_sequence.device)


def _unique_speaker_sequence_candidates(
    speech_sequence: torch.Tensor,
    speech_mask: torch.Tensor,
    label_id: torch.Tensor,
    speaker_ids: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Keep one sequence target for every (speaker, text) pair."""
    if speech_sequence.shape[0] != len(speaker_ids):
        raise ValueError("speaker_ids must contain one value per speech sequence")

    first_by_key: dict[tuple[str, int], int] = {}
    for index, (speaker_id, text_id) in enumerate(zip(speaker_ids, label_id.detach().cpu().tolist())):
        first_by_key.setdefault((str(speaker_id), int(text_id)), index)

    ordered_keys = sorted(first_by_key)
    candidate_indices = torch.tensor(
        [first_by_key[key] for key in ordered_keys],
        dtype=torch.long,
        device=speech_sequence.device,
    )
    return (
        speech_sequence.index_select(0, candidate_indices),
        speech_mask.index_select(0, candidate_indices),
        torch.tensor(
            [text_id for _, text_id in ordered_keys],
            dtype=label_id.dtype,
            device=label_id.device,
        ),
        [speaker_id for speaker_id, _ in ordered_keys],
    )


def _sequence_retrieval_topk_against_candidates(
    eeg_sequence: torch.Tensor,
    eeg_mask: torch.Tensor,
    query_label_id: torch.Tensor,
    candidate_sequence: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_label_id: torch.Tensor,
    k: int,
    chunk_size: int = 256,
    alignment_max_lag: int = 0,
    alignment_min_frames: int = 1,
) -> torch.Tensor:
    if eeg_sequence.shape[0] == 0:
        return torch.tensor(0.0, device=eeg_sequence.device)

    total_correct = 0.0
    total = 0
    for start in range(0, eeg_sequence.shape[0], chunk_size):
        stop = min(start + chunk_size, eeg_sequence.shape[0])
        logits = sequence_similarity_logits(
            eeg_sequence[start:stop],
            candidate_sequence,
            eeg_mask[start:stop],
            candidate_mask,
            temperature=1.0,
            alignment_max_lag=alignment_max_lag,
            alignment_min_frames=alignment_min_frames,
        )
        predictions = logits.topk(min(k, logits.shape[1]), dim=1).indices
        total_correct += float(
            candidate_label_id[predictions].eq(query_label_id[start:stop].unsqueeze(1)).any(dim=1).float().sum()
        )
        total += stop - start
    return torch.tensor(total_correct / max(total, 1), device=eeg_sequence.device)


def speaker_full_sequence_retrieval_topk(
    eeg_sequence: torch.Tensor,
    speech_sequence: torch.Tensor,
    eeg_mask: torch.Tensor,
    speech_mask: torch.Tensor,
    label_id: torch.Tensor,
    speaker_ids: list[str],
    k: int,
    alignment_max_lag: int = 0,
    alignment_min_frames: int = 1,
) -> torch.Tensor:
    """Retrieve against only the targets narrated by the query's speaker."""
    candidates, candidate_masks, candidate_label_ids, candidate_speaker_ids = _unique_speaker_sequence_candidates(
        speech_sequence,
        speech_mask,
        label_id,
        speaker_ids,
    )
    total_correct = 0.0
    total = 0
    for speaker_id in sorted(set(speaker_ids)):
        query_indices = torch.tensor(
            [index for index, value in enumerate(speaker_ids) if value == speaker_id],
            dtype=torch.long,
            device=eeg_sequence.device,
        )
        candidate_indices = torch.tensor(
            [index for index, value in enumerate(candidate_speaker_ids) if value == speaker_id],
            dtype=torch.long,
            device=eeg_sequence.device,
        )
        accuracy = _sequence_retrieval_topk_against_candidates(
            eeg_sequence.index_select(0, query_indices),
            eeg_mask.index_select(0, query_indices),
            label_id.index_select(0, query_indices),
            candidates.index_select(0, candidate_indices),
            candidate_masks.index_select(0, candidate_indices),
            candidate_label_ids.index_select(0, candidate_indices),
            k,
            alignment_max_lag=alignment_max_lag,
            alignment_min_frames=alignment_min_frames,
        )
        total_correct += float(accuracy) * len(query_indices)
        total += len(query_indices)
    return torch.tensor(total_correct / max(total, 1), device=eeg_sequence.device)


def dual_positive_full_sequence_retrieval_topk(
    eeg_sequence: torch.Tensor,
    speech_sequence: torch.Tensor,
    eeg_mask: torch.Tensor,
    speech_mask: torch.Tensor,
    label_id: torch.Tensor,
    speaker_ids: list[str],
    k: int,
    alignment_max_lag: int = 0,
    alignment_min_frames: int = 1,
) -> torch.Tensor:
    """Retrieve over both narrators; either narrator of the correct text is positive."""
    candidates, candidate_masks, candidate_label_ids, _ = _unique_speaker_sequence_candidates(
        speech_sequence,
        speech_mask,
        label_id,
        speaker_ids,
    )
    return _sequence_retrieval_topk_against_candidates(
        eeg_sequence,
        eeg_mask,
        label_id,
        candidates,
        candidate_masks,
        candidate_label_ids,
        k,
        alignment_max_lag=alignment_max_lag,
        alignment_min_frames=alignment_min_frames,
    )


def run_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    temperature: float,
    alignment_max_lag: int = 0,
    alignment_min_frames: int = 1,
) -> tuple[float, float, float, float]:
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
        loss, logits = eeg_to_speech_sequence_loss(
            pred,
            speech,
            pred_mask,
            speech_mask,
            label_id,
            temperature,
            alignment_max_lag=alignment_max_lag,
            alignment_min_frames=alignment_min_frames,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = eeg.shape[0]
        total += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_cos += float(
            sequence_cosine_mean(
                pred.detach(),
                speech,
                pred_mask,
                speech_mask,
                alignment_max_lag=alignment_max_lag,
                alignment_min_frames=alignment_min_frames,
            )
        ) * batch_size
        total_top1 += float(sequence_retrieval_topk(logits.detach(), label_id, k=1)) * batch_size
        total_top10 += float(sequence_retrieval_topk(logits.detach(), label_id, k=10)) * batch_size
    return total_loss / total, total_cos / total, total_top1 / total, total_top10 / total


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    temperature: float,
    include_speaker_protocols: bool = False,
    alignment_max_lag: int = 0,
    alignment_min_frames: int = 1,
) -> dict[str, float]:
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
    all_speaker_ids: list[str] = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        eeg_mask = batch["eeg_mask"].to(device)
        speech = batch["speech"].to(device)
        speech_mask = batch["speech_mask"].to(device)
        label_id = batch["label_id"].to(device)
        pred, pred_mask = model(eeg, eeg_mask, return_mask=True)
        loss, logits = eeg_to_speech_sequence_loss(
            pred,
            speech,
            pred_mask,
            speech_mask,
            label_id,
            temperature,
            alignment_max_lag=alignment_max_lag,
            alignment_min_frames=alignment_min_frames,
        )
        batch_size = eeg.shape[0]
        total += batch_size
        total_loss += float(loss) * batch_size
        total_cos += float(
            sequence_cosine_mean(
                pred,
                speech,
                pred_mask,
                speech_mask,
                alignment_max_lag=alignment_max_lag,
                alignment_min_frames=alignment_min_frames,
            )
        ) * batch_size
        total_top1 += float(sequence_retrieval_topk(logits, label_id, k=1)) * batch_size
        total_top10 += float(sequence_retrieval_topk(logits, label_id, k=10)) * batch_size
        all_pred.append(pred.detach().cpu())
        all_pred_mask.append(pred_mask.detach().cpu())
        all_speech.append(speech.detach().cpu())
        all_speech_mask.append(speech_mask.detach().cpu())
        all_label_id.append(label_id.detach().cpu())
        all_speaker_ids.extend(str(meta["speaker_id"]) for meta in batch["meta"])

    pred_all = torch.cat(all_pred, dim=0)
    pred_mask_all = torch.cat(all_pred_mask, dim=0)
    max_speech_frames = max(sequence.shape[1] for sequence in all_speech)
    speech_all = torch.cat(
        [F.pad(sequence, (0, 0, 0, max_speech_frames - sequence.shape[1])) for sequence in all_speech],
        dim=0,
    )
    speech_mask_all = torch.cat(
        [F.pad(mask, (0, max_speech_frames - mask.shape[1])) for mask in all_speech_mask],
        dim=0,
    )
    label_id_all = torch.cat(all_label_id, dim=0)
    metrics = {
        "loss": total_loss / total,
        "cos": total_cos / total,
        "top1": total_top1 / total,
        "top10": total_top10 / total,
        # Kept for backward compatibility. Its single candidate is the first
        # observed narrator for each text and should not be the primary report.
        "full_top1": float(
            full_sequence_retrieval_topk(
                pred_all,
                speech_all,
                pred_mask_all,
                speech_mask_all,
                label_id_all,
                k=1,
                alignment_max_lag=alignment_max_lag,
                alignment_min_frames=alignment_min_frames,
            )
        ),
        "full_top10": float(
            full_sequence_retrieval_topk(
                pred_all,
                speech_all,
                pred_mask_all,
                speech_mask_all,
                label_id_all,
                k=10,
                alignment_max_lag=alignment_max_lag,
                alignment_min_frames=alignment_min_frames,
            )
        ),
    }
    if include_speaker_protocols:
        metrics.update(
            {
                "speaker_full_top1": float(
                    speaker_full_sequence_retrieval_topk(
                        pred_all,
                        speech_all,
                        pred_mask_all,
                        speech_mask_all,
                        label_id_all,
                        all_speaker_ids,
                        k=1,
                        alignment_max_lag=alignment_max_lag,
                        alignment_min_frames=alignment_min_frames,
                    )
                ),
                "speaker_full_top10": float(
                    speaker_full_sequence_retrieval_topk(
                        pred_all,
                        speech_all,
                        pred_mask_all,
                        speech_mask_all,
                        label_id_all,
                        all_speaker_ids,
                        k=10,
                        alignment_max_lag=alignment_max_lag,
                        alignment_min_frames=alignment_min_frames,
                    )
                ),
                "dual_full_top1": float(
                    dual_positive_full_sequence_retrieval_topk(
                        pred_all,
                        speech_all,
                        pred_mask_all,
                        speech_mask_all,
                        label_id_all,
                        all_speaker_ids,
                        k=1,
                        alignment_max_lag=alignment_max_lag,
                        alignment_min_frames=alignment_min_frames,
                    )
                ),
                "dual_full_top10": float(
                    dual_positive_full_sequence_retrieval_topk(
                        pred_all,
                        speech_all,
                        pred_mask_all,
                        speech_mask_all,
                        label_id_all,
                        all_speaker_ids,
                        k=10,
                        alignment_max_lag=alignment_max_lag,
                        alignment_min_frames=alignment_min_frames,
                    )
                ),
            }
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sequence-frames", type=int, default=None)
    parser.add_argument("--alignment-max-lag", type=int, default=None)
    parser.add_argument("--alignment-min-frames", type=int, default=None)
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
    if args.sequence_frames is not None:
        sequence_frames_value: int | None = args.sequence_frames
    elif "sequence_frames" in config:
        sequence_frames_value = config["sequence_frames"]
    elif "sequence_frames" in model_config:
        sequence_frames_value = model_config["sequence_frames"]
    else:
        sequence_frames_value = 64
    sequence_frames = None if sequence_frames_value is None or int(sequence_frames_value) <= 0 else int(sequence_frames_value)
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
    alignment_max_lag = int(
        coalesce(
            args.alignment_max_lag,
            nested_get(config, "train", "alignment_max_lag"),
            config.get("alignment_max_lag"),
            0,
        )
    )
    alignment_min_frames = int(
        coalesce(
            args.alignment_min_frames,
            nested_get(config, "train", "alignment_min_frames"),
            config.get("alignment_min_frames"),
            1,
        )
    )
    if alignment_max_lag < 0:
        raise ValueError(f"alignment_max_lag must be non-negative, got {alignment_max_lag}")
    if alignment_min_frames <= 0:
        raise ValueError(f"alignment_min_frames must be positive, got {alignment_min_frames}")
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
    if checkpoint_metric not in SEQUENCE_METRIC_MODES:
        raise ValueError(
            f"checkpoint_metric must be one of {sorted(SEQUENCE_METRIC_MODES)}, got {checkpoint_metric!r}"
        )
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
        f"sequence_frames={sequence_frames if sequence_frames is not None else 'variable'} "
        f"alignment_max_lag={alignment_max_lag} alignment_min_frames={alignment_min_frames} "
        f"unique_text_per_batch={unique_text_per_batch}"
    )

    device = torch.device(device_name)
    model = TemporalConvEEGSequenceEncoder(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_mode = SEQUENCE_METRIC_MODES[checkpoint_metric]
    checkpoint_uses_speaker_protocol = checkpoint_metric.startswith(("val_speaker_", "val_dual_"))
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
            alignment_max_lag=alignment_max_lag,
            alignment_min_frames=alignment_min_frames,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            contrastive_temperature,
            include_speaker_protocols=checkpoint_uses_speaker_protocol,
            alignment_max_lag=alignment_max_lag,
            alignment_min_frames=alignment_min_frames,
        )
        current_metrics = {
            "val_loss": val_metrics["loss"],
            "val_cos": val_metrics["cos"],
            "val_top1": val_metrics["top1"],
            "val_top10": val_metrics["top10"],
            "val_full_top1": val_metrics["full_top1"],
            "val_full_top10": val_metrics["full_top10"],
        }
        if checkpoint_uses_speaker_protocol:
            current_metrics.update(
                {
                    "val_speaker_full_top1": val_metrics["speaker_full_top1"],
                    "val_speaker_full_top10": val_metrics["speaker_full_top10"],
                    "val_dual_full_top1": val_metrics["dual_full_top1"],
                    "val_dual_full_top10": val_metrics["dual_full_top10"],
                }
            )
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
                    "alignment_max_lag": alignment_max_lag,
                    "alignment_min_frames": alignment_min_frames,
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
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            contrastive_temperature,
            include_speaker_protocols=True,
            alignment_max_lag=alignment_max_lag,
            alignment_min_frames=alignment_min_frames,
        )
        print(
            f"test_loss={test_metrics['loss']:.5f} test_cos={test_metrics['cos']:.4f} "
            f"test_top1={test_metrics['top1']:.4f} test_top10={test_metrics['top10']:.4f} "
            f"test_full_top1={test_metrics['full_top1']:.4f} test_full_top10={test_metrics['full_top10']:.4f} "
            f"test_speaker_full_top1={test_metrics['speaker_full_top1']:.4f} "
            f"test_speaker_full_top10={test_metrics['speaker_full_top10']:.4f} "
            f"test_dual_full_top1={test_metrics['dual_full_top1']:.4f} "
            f"test_dual_full_top10={test_metrics['dual_full_top10']:.4f}"
        )


if __name__ == "__main__":
    main()
