from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chineseeeg2_littleprince.data import EEGTextDataset, collate_eeg_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=1300)
    args = parser.parse_args()

    dataset = EEGTextDataset(args.manifest)
    print(f"dataset size: {len(dataset)}")

    first = dataset[0]
    print(f"first eeg: {tuple(first['eeg'].shape)}")
    print(f"first label: {tuple(first['label'].shape)}")
    print(f"first meta: {first['meta']}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_eeg_text, max_samples=args.max_samples),
    )
    batch = next(iter(loader))
    print(f"batch eeg: {tuple(batch['eeg'].shape)}")
    print(f"batch label: {tuple(batch['label'].shape)}")
    print(f"batch mask: {tuple(batch['mask'].shape)}")
    print(f"batch length: {batch['length'].tolist()}")


if __name__ == "__main__":
    main()
