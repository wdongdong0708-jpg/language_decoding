# ChineseEEG-2 LittlePrince PL Project

This project is a minimal PyTorch-style pipeline for line-level EEG-text alignment on ChineseEEG-2:

- Task: Passive Listening
- Subjects: clean Little Prince passive-listening runs from available subjects
- Session: ses-littleprince
- EEG source: preprocessed BrainVision files
- Label source: BERT text embeddings for The Little Prince

The all-clean manifest keeps only runs that pass strict line-level ROWS/ROWE alignment checks.

## Project Layout

```text
chineseeeg2_littleprince_pl/
  configs/
    all_clean.yaml
  data/
    manifests/
      littleprince_pl_all_clean_manifest.csv
      littleprince_pl_sub08_manifest.csv
  scripts/
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
y = text_embeddings_littleprince[text_embedding_idx]
```

EEG windows are variable length. The DataLoader collate function pads or crops them into a fixed batch tensor:

```text
eeg:   [batch, 128, time]
label: [batch, 768]
mask:  [batch, time]
text_embedding_idx: [batch]
```

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

Command-line arguments can still override the config, for example:

```bash
python -m chineseeeg2_littleprince.train --config configs/all_clean.yaml --epochs 20 --batch-size 64
```

The baseline is intentionally small: a temporal Conv1D encoder with masked mean pooling and a 768-dimensional projection head.
It trains with an EEG-to-text in-batch contrastive loss over cosine similarities. Train/validation/test splits are grouped by `text_embedding_idx`, the default batch sampler avoids duplicate text IDs within a batch, and the loss/top-k metrics support multi-positive labels when duplicates are present.
Training saves the best checkpoint by `val_full_top10`, uses early stopping, and reports both batch-level retrieval (`val_top1/top10`) and full-split retrieval (`val_full_top1/top10`, `test_full_top1/top10`).
