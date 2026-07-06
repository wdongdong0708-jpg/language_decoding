# ChineseEEG-2 LittlePrince PL Project

This project is a minimal PyTorch-style pipeline for line-level EEG-text alignment on ChineseEEG-2:

- Task: Passive Listening
- Subject: sub-08
- Session: ses-littleprince
- EEG source: preprocessed BrainVision files
- Label source: BERT text embeddings for The Little Prince

The first version intentionally starts with only `sub-08`, because all 27 Little Prince runs pass strict `ROWS/ROWE` alignment checks.

## Project Layout

```text
chineseeeg2_littleprince_pl/
  configs/
    sub08.yaml
  data/
    manifests/
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
```

## Setup

Use a Python environment with PyTorch installed correctly.

```bash
pip install -r requirements.txt
pip install -e .
```

The current machine has a broken Torch DLL import, so the non-torch reader can be tested now, but Dataset/DataLoader execution needs the Torch install repaired first.

## Smoke Check

After Torch works:

```bash
python scripts/smoke_check.py --manifest data/manifests/littleprince_pl_sub08_manifest.csv --batch-size 4 --max-samples 1300
```

Expected output:

```text
dataset size: 2837
eeg: torch.Size([4, 128, 1300])
label: torch.Size([4, 768])
mask: torch.Size([4, 1300])
```

## Train Baseline

```bash
python -m chineseeeg2_littleprince.train --config configs/sub08.yaml
```

Command-line arguments can still override the config, for example:

```bash
python -m chineseeeg2_littleprince.train --config configs/sub08.yaml --epochs 20 --batch-size 8
```

The baseline is intentionally small: a temporal Conv1D encoder with masked mean pooling and a 768-dimensional projection head.
It trains with an EEG-to-text in-batch contrastive loss over cosine similarities.
