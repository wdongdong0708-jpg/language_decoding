from __future__ import annotations

import argparse
import csv
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chineseeeg2_littleprince.data import EEGSpeechSequenceDataset, collate_eeg_speech_sequence  # noqa: E402
from chineseeeg2_littleprince.data.sequence_collate import _fit_feature_sequence  # noqa: E402
from chineseeeg2_littleprince.models import TemporalConvEEGSequenceEncoder  # noqa: E402
from chineseeeg2_littleprince.train import coalesce, load_config, nested_get, resolve_manifest_path  # noqa: E402
from chineseeeg2_littleprince.train_speech_sequence import sequence_similarity_logits  # noqa: E402


FIELDNAMES = [
    "query_index",
    "rank",
    "score",
    "hit",
    "query_subject",
    "query_text_embedding_idx",
    "query_text",
    "candidate_speaker_id",
    "candidate_speech_sequence_idx",
    "candidate_text_embedding_idx",
    "candidate_text",
    "candidate_audio_file_path",
    "candidate_audio_start_time",
    "candidate_audio_stop_time",
]


def candidate_table(
    dataset: EEGSpeechSequenceDataset,
    sequence_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    first_by_sequence_idx = {}
    for record in dataset.records:
        key = (str(record.speech_sequence_path), record.speech_sequence_idx)
        first_by_sequence_idx.setdefault(key, record)

    cache: dict[Path, np.ndarray] = {}
    sequences = []
    masks = []
    metas = []
    for _, record in sorted(first_by_sequence_idx.items(), key=lambda item: item[1].speech_sequence_idx):
        if record.speech_sequence_path not in cache:
            cache[record.speech_sequence_path] = np.load(record.speech_sequence_path, mmap_mode="r")
        array = cache[record.speech_sequence_path][record.speech_frame_start : record.speech_frame_stop]
        resized, mask, _ = _fit_feature_sequence(torch.from_numpy(np.asarray(array, dtype=np.float32)), sequence_frames)
        sequences.append(resized)
        masks.append(mask)
        metas.append(
            {
                "label_id": record.label_id,
                "speaker_id": record.speaker_id,
                "speech_sequence_idx": record.speech_sequence_idx,
                "text_embedding_idx": record.eeg.text_embedding_idx,
                "text": record.text,
                "audio_file_path": str(record.audio_file_path),
                "audio_start_time": record.audio_start_time,
                "audio_stop_time": record.audio_stop_time,
            }
        )
    return torch.stack(sequences, dim=0), torch.stack(masks, dim=0), metas


def write_rows(path: Path | None, rows: list[dict[str, object]]) -> None:
    if path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/littleprince_sentence_audio_sequence.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sequence-frames", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config.resolve() if args.config is not None else None
    config = load_config(config_path)
    model_config = dict(config.get("model", {}))
    manifest = coalesce(args.manifest, config.get("manifest"))
    if manifest is None:
        parser.error("one of --config with a manifest field or --manifest is required")

    checkpoint_path = Path(
        coalesce(
            args.checkpoint,
            nested_get(config, "train", "checkpoint_path"),
            config.get("checkpoint_path"),
        )
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path

    max_samples = int(coalesce(args.max_samples, config.get("max_samples"), 1300))
    sequence_frames = int(
        coalesce(args.sequence_frames, config.get("sequence_frames"), model_config.get("sequence_frames"), 64)
    )
    device = torch.device(args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = EEGSpeechSequenceDataset(resolve_manifest_path(manifest, config_path))
    query_indices = list(range(len(dataset)))
    if args.max_queries is not None and args.max_queries > 0:
        query_indices = query_indices[: args.max_queries]
    loader = DataLoader(
        Subset(dataset, query_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_eeg_speech_sequence,
            max_samples=max_samples,
            sequence_frames=sequence_frames,
        ),
    )

    model_kwargs = {
        "in_channels": 128,
        "hidden_channels": 128,
        "embedding_dim": 1024,
        "sequence_frames": sequence_frames,
        "dropout": 0.2,
    }
    model_kwargs.update(model_config)
    model_kwargs["sequence_frames"] = sequence_frames
    model = TemporalConvEEGSequenceEncoder(**model_kwargs).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    candidate_sequences, candidate_masks, candidate_metas = candidate_table(dataset, sequence_frames)
    candidate_sequences = candidate_sequences.to(device)
    candidate_masks = candidate_masks.to(device)
    top_k = min(args.top_k, candidate_sequences.shape[0])

    output_rows = []
    query_cursor = 0
    for batch in loader:
        eeg = batch["eeg"].to(device)
        eeg_mask = batch["eeg_mask"].to(device)
        label_id = batch["label_id"].to(device)
        pred, pred_mask = model(eeg, eeg_mask, return_mask=True)
        logits = sequence_similarity_logits(
            pred,
            candidate_sequences,
            pred_mask,
            candidate_masks,
            temperature=1.0,
        )
        scores, indices = logits.topk(top_k, dim=1)
        for batch_index, meta in enumerate(batch["meta"]):
            query_index = query_indices[query_cursor + batch_index]
            for rank in range(top_k):
                candidate_index = int(indices[batch_index, rank])
                candidate = candidate_metas[candidate_index]
                output_rows.append(
                    {
                        "query_index": query_index,
                        "rank": rank + 1,
                        "score": f"{float(scores[batch_index, rank]):.6f}",
                        "hit": int(candidate["label_id"] == int(label_id[batch_index])),
                        "query_subject": meta["subject"],
                        "query_text_embedding_idx": meta["text_embedding_idx"],
                        "query_text": meta.get("text", ""),
                        "candidate_speaker_id": candidate["speaker_id"],
                        "candidate_speech_sequence_idx": candidate["speech_sequence_idx"],
                        "candidate_text_embedding_idx": candidate["text_embedding_idx"],
                        "candidate_text": candidate["text"],
                        "candidate_audio_file_path": candidate["audio_file_path"],
                        "candidate_audio_start_time": f"{candidate['audio_start_time']:.6f}",
                        "candidate_audio_stop_time": f"{candidate['audio_stop_time']:.6f}",
                    }
                )
        query_cursor += len(batch["meta"])

    write_rows(args.output, output_rows)


if __name__ == "__main__":
    main()
