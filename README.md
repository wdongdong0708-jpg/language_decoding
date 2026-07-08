# ChineseEEG-2 PL EEG-Text Project

This project is a minimal PyTorch-style pipeline for line-level EEG-text alignment on ChineseEEG-2:

- Task: Passive Listening
- Subjects: clean passive-listening runs from available subjects
- Sessions: ses-littleprince and supported ses-garnettdream runs
- EEG source: preprocessed BrainVision files
- Label source: BERT text embeddings for The Little Prince and Garnett Dream

The all-clean manifests keep only runs that pass strict line-level ROWS/ROWE alignment checks.

## Project Layout

```text
chineseeeg2_littleprince_pl/
  configs/
    all_clean.yaml
    garnettdream.yaml
    pl_littleprince_garnettdream.yaml
  data/
    manifests/
      littleprince_pl_all_clean_manifest.csv
      littleprince_pl_sub08_manifest.csv
  scripts/
    build_garnettdream_pl_manifest.py
    smoke_check.py
  src/
    chineseeeg2_littleprince/
      data/
        collate.py
        dataset.py
        manifest.py
      io/
        brainvision.py
      models/
        baseline.py
      train.py
  tests/
    test_manifest_and_reader.py
```

## Data Definition

Each row in the manifest is one sample:

```text
X = EEG[:, start_sample:stop_sample]
y = text_embedding_path[text_embedding_idx]
```

EEG windows are variable length. The DataLoader collate function pads or crops them into a fixed batch tensor:

```text
eeg:   [batch, 128, time]
label: [batch, 768]
mask:  [batch, time]
text_embedding_idx: [batch]
label_id: [batch]
```

`label_id` is used only for split/sampler/loss grouping. This matters for mixed-novel training because
`text_embeddings_littleprince.npy` and `text_embeddings_garnettdream.npy` both have local row indices.

## Setup

Use a Python environment with PyTorch installed correctly.

```bash
pip install -r requirements.txt
pip install -e .
```

## Smoke Check

After Torch works:

```bash
python scripts/smoke_check.py --manifest data/manifests/littleprince_pl_all_clean_manifest.csv --batch-size 4 --max-samples 1300
```

Expected output:

```text
dataset size: 21110
eeg: torch.Size([4, 128, 1300])
label: torch.Size([4, 768])
mask: torch.Size([4, 1300])
```

## Train Baseline

```bash
python -m chineseeeg2_littleprince.train --config configs/all_clean.yaml
```

## Sentence-Level Speech Retrieval

Build a Little Prince passive-listening manifest that points each EEG row to the matching sentence-level audio window:

```bash
python scripts/build_littleprince_sentence_audio_manifest.py --skip-embeddings
```

Then generate sentence-level audio embeddings with the official speech-model recipe:

```bash
python scripts/build_littleprince_sentence_audio_manifest.py --overwrite
```

Train EEG-to-speech retrieval after the sentence-level audio embedding file exists:

```bash
python -m chineseeeg2_littleprince.train_speech --config configs/littleprince_sentence_audio.yaml
```

Inspect top retrieval results from a trained checkpoint:

```bash
python scripts/retrieve_sentence_audio.py --config configs/littleprince_sentence_audio.yaml --output data/manifests/littleprince_sentence_audio_retrieval.csv
```

The generated speech manifest keeps the original EEG fields and adds speaker, audio file, audio window, and speech embedding columns.

## Add Garnett Dream

Build the supported Garnett Dream PL manifests:

```bash
python scripts/build_garnettdream_pl_manifest.py
```

This writes:

```text
data/manifests/garnettdream_pl_all_clean_manifest.csv
data/manifests/garnettdream_alignment_report.csv
data/manifests/pl_littleprince_garnettdream_all_clean_manifest.csv
```

The provided `text_embeddings_garnettdream.npy` has shape `(2164, 768)`, which matches the 9-run PL layout:
17 preface rows plus 2147 formal ROWS/ROWE windows. On the local dataset this safely includes `sub-01` to
`sub-04`. The 8-run Garnett Dream layout in `sub-05` to `sub-08` is reported and skipped because it has 2491
formal ROWS/ROWE windows and needs a matching text embedding file.

Train only Garnett Dream:

```bash
python -m chineseeeg2_littleprince.train --config configs/garnettdream.yaml
```

Train the mixed Little Prince + Garnett Dream manifest:

```bash
python -m chineseeeg2_littleprince.train --config configs/pl_littleprince_garnettdream.yaml
```

Command-line arguments can still override the config, for example:

```bash
python -m chineseeeg2_littleprince.train --config configs/all_clean.yaml --epochs 20 --batch-size 64
```

The baseline is intentionally small: a temporal Conv1D encoder with masked mean pooling and a 768-dimensional projection head.
It trains with an EEG-to-text in-batch contrastive loss over cosine similarities. Train/validation/test splits are grouped by `text_embedding_idx`, the default batch sampler avoids duplicate text IDs within a batch, and the loss/top-k metrics support multi-positive labels when duplicates are present.
Training saves the best checkpoint by `val_full_top10`, uses early stopping, and reports both batch-level retrieval (`val_top1/top10`) and full-split retrieval (`val_full_top1/top10`, `test_full_top1/top10`).
