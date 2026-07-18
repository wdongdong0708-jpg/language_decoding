# ChineseEEG-2 EEG-Text Project

This project is a minimal PyTorch-style pipeline for line-level EEG-text alignment on ChineseEEG-2:

- Tasks: Passive Listening and Reading Aloud
- Subjects: clean runs from the available subjects for each task
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

At load time, legacy manifests are enriched with three separate identities:

- `instance_id`: one subject/run/row EEG observation.
- `target_id` / `target_uid`: the canonical text target, generated from the exact float32 embedding SHA256.
- `split_group_id`: the unit assigned to train/validation/test; it defaults to the canonical target UID.

This separation prevents identical embeddings at different occurrence rows from becoming false negatives or
appearing in multiple splits. Explicit identity columns in a future manifest override these defaults.

EEG windows are variable length. The DataLoader collate function pads or crops them into a fixed batch tensor:

```text
eeg:   [batch, 128, time]
label: [batch, 768]
mask:  [batch, time]
text_embedding_idx: [batch]
label_id: [batch]
target_id: [batch]
subject_id: [batch]
```

`label_id` is retained as the legacy occurrence-row label. Split, sampler, loss, and retrieval semantics use
`split_group_id` and `target_id` instead. This matters both for repeated lines and for mixed-novel training.

Splits follow Meta's deterministic protocol: `SHA256(split_group_id)` plus the configured seed assigns complete
groups to an 80/10/10 train/validation/test partition. Adding unrelated targets does not reshuffle existing ones.

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

## Reading Aloud

The Passive Listening manifests read EEG and events from:

```text
D:\dataset\ChineseEEG-2\PassiveListening\derivatives\preprocessed
```

Reading Aloud needs its own manifest because its subjects, task name, EEG/event paths, and ROWS/ROWE
time windows differ from Passive Listening. The text embeddings and their per-run indices are shared.
The builder therefore derives the line-to-embedding mapping from the validated Passive Listening manifest,
then applies it to strictly aligned Reading Aloud runs:

```bash
conda run -n bm5060 python scripts/build_readingaloud_manifest.py
```

This writes:

```text
data/manifests/littleprince_ra_all_clean_manifest.csv
data/manifests/littleprince_ra_alignment_report.csv
```

Switch to Reading Aloud by selecting its config; the training code and dataset class stay unchanged:

```bash
conda run -n bm5060 python -m chineseeeg2_littleprince.train --config configs/reading_aloud.yaml
```

The same builder can use another Passive Listening manifest as its reference. For example, pass
`--reference-manifest data/manifests/garnettdream_pl_all_clean_manifest.csv` together with distinct
`--output-manifest` and `--alignment-report` paths to build the supported Reading Aloud Garnett Dream runs.

### Four-subject Passive Listening comparison

To compare Passive Listening and Reading Aloud with the same number of subjects, use the PL config
that selects `sub-01` through `sub-04` from the unchanged all-clean manifest:

```bash
conda run -n bm5060 python -m chineseeeg2_littleprince.train --config configs/passive_listening_4subjects.yaml
```

The optional top-level `subjects` list filters a manifest before identities, splits, and subject IDs are
constructed. This four-subject subset has 10,390 rows and still covers all 2,603 canonical targets.

## Silent Reading from ChineseEEG

ChineseEEG is a separate dataset from ChineseEEG-2. Its filtered BrainVision recordings are under
`D:\dataset\ChineseEEG\filtered_0.5_30`, use 256 Hz sampling, organize Little Prince as runs 01–07,
and store one `(rows, 768)` text embedding array per run. The manifest builder excludes any practice
ROWS/ROWE markers before the first chapter marker and validates every formal event pair against the
matching per-run embedding array:

```bash
conda run -n bm5060 python scripts/build_readingsilent_manifest.py
```

The default builds the Little Prince manifest used for direct PL/RA/ReadingSilent comparisons. Use
`--corpus garnettdream` or `--corpus all` for the other supported catalogs. Little Prince contains
all seven runs for eight subjects and runs 04–07 for `sub-09`; Garnett Dream run 19 is skipped when
no corresponding text embedding exists. Train Silent Reading with:

```bash
conda run -n bm5060 python -m chineseeeg2_littleprince.train --config configs/reading_silent.yaml
```

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

P1 uses a compact `SimpleConvTimeAgg` while retaining the original `temporal_conv` model for ablations. The compact
model applies a shared 1x1 channel projection, an identity-initialized subject-specific linear layer, input dropout,
five residual Conv1D blocks with dilations `1/2/4/8/16`, and contextual GLUs after blocks 2 and 4. Masked Bahdanau
attention aggregates time before the 768-dimensional projection head. Set `model.name: temporal_conv` to run the old
baseline. Both EEG encoders additionally learn a two-layer text projection head immediately before cosine-similarity
loss and retrieval scoring; `text_embedding_dim` and `text_projection_hidden_dim` configure it.

Training uses an EEG-to-text in-batch contrastive loss over cosine similarities. The default batch sampler avoids
duplicate canonical targets within a batch, while the loss still supports multi-positive targets. For BatchNorm
stability, training filters unique-target batches smaller than `train.min_train_batch_size` (default 32) and shuffles
the completed batch list; validation and test retain every row.

Training saves the best checkpoint by `val_full_macro_top10`, uses early stopping, and reports:

- batch-level Top-1/Top-10;
- full-vocabulary and most-frequent-250 Top-1/Top-10;
- target-macro Top-1/Top-10;
- mean/median rank;
- instance-aggregated retrieval after averaging repeated predictions of the same target.

The checkpoint stores the exact identity and split protocol, including all split-group UIDs.
