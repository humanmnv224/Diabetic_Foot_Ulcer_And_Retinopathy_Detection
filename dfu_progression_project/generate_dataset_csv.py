import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
DEFAULT_CLASS_TO_LABEL = {
    "normal": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}


def collect_rows(split_dir: Path, class_to_label: Dict[str, int]) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []

    for class_name, label in class_to_label.items():
        class_dir = split_dir / class_name
        if not class_dir.exists() or not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        files = [
            p.resolve()
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
        ]

        if not files:
            raise ValueError(f"No images found in class folder: {class_dir}")

        rows.extend((str(path), label) for path in sorted(files))

    return rows


def write_csv(rows: List[Tuple[str, int]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["image_path", "label"])
    df.to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DFU Lightning CSV files from dataset folders.")
    parser.add_argument("--dataset_root", default="dataset", help="Path containing train/ and test/ folders")
    parser.add_argument("--train_split", default="train", help="Name of train split folder")
    parser.add_argument("--val_split", default="test", help="Name of validation split folder")
    parser.add_argument("--train_csv", default="dataset/dfu_train.csv", help="Output train CSV path")
    parser.add_argument("--val_csv", default="dataset/dfu_val.csv", help="Output val CSV path")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    train_dir = dataset_root / args.train_split
    val_dir = dataset_root / args.val_split

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    train_rows = collect_rows(train_dir, DEFAULT_CLASS_TO_LABEL)
    val_rows = collect_rows(val_dir, DEFAULT_CLASS_TO_LABEL)

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)

    write_csv(train_rows, train_csv)
    write_csv(val_rows, val_csv)

    print(f"Saved train CSV: {train_csv} ({len(train_rows)} rows)")
    print(f"Saved val CSV:   {val_csv} ({len(val_rows)} rows)")


if __name__ == "__main__":
    main()
