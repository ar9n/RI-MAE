import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/EngineeringShapeBenchmark")
N_POINTS = 8192


def create_train_test_split_per_subfolder(
    data_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """Create an 80/20 split separately for each subfolder in data_dir.

    Also creates a class-to-index mapping where class is the parent folder
    name of each STL file, and indices start at 0.
    """
    rng = np.random.default_rng(seed)
    subfolder_files = {}

    stl_files = sorted(
        f for f in data_dir.rglob("*")
        if f.is_file() and f.suffix.lower() == ".stl"
    )

    for stl_file in stl_files:
        rel_stl = stl_file.relative_to(data_dir)
        folder_key = rel_stl.parent.as_posix()
        subfolder_files.setdefault(folder_key, []).append(rel_stl.as_posix())

    if not subfolder_files:
        print("No .stl files found for per-subfolder split")
        return {}, [], []

    train_files = []
    test_files = []
    label_map = {}

    for folder_key in sorted(subfolder_files.keys()):
        files = sorted(subfolder_files[folder_key])
        rng.shuffle(files)

        if len(files) == 1:
            split_idx = 1
        else:
            split_idx = int(len(files) * train_ratio)
            split_idx = max(1, min(split_idx, len(files) - 1))

        train_files.extend(files[:split_idx])
        test_files.extend(files[split_idx:])

        print(
            f"Folder '{folder_key}': total={len(files)}, "
            f"train={len(files[:split_idx])}, test={len(files[split_idx:])}"
        )

        # assign a unique label index for this folder
        label_map[folder_key] = len(label_map)

    train_txt = data_dir / "train.txt"
    test_txt = data_dir / "test.txt"
    label_map_json = data_dir / "label_map.json"

    with open(train_txt, "w") as f:
        f.write("\n".join(train_files))

    with open(test_txt, "w") as f:
        f.write("\n".join(test_files))

    with open(label_map_json, "w") as f:
        json.dump(label_map, f, indent=2)

    print("Per-subfolder train/test split created:")
    print(f"  Train: {len(train_files)} files -> {train_txt}")
    print(f"  Test: {len(test_files)} files -> {test_txt}")
    print(f"  Classes: {len(label_map)} -> {label_map_json}")

    return label_map, train_files, test_files


if __name__ == "__main__":
    create_train_test_split_per_subfolder(DATA_DIR)
