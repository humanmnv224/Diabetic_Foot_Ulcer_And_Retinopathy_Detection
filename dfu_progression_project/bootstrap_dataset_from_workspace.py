import argparse
import shutil
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import train_test_split

from config import DATA_ROOT, SEED

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def collect_images(folder: Path) -> List[Path]:
    """Collect image files recursively from a folder."""
    if not folder.exists() or not folder.is_dir():
        return []

    images: List[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
            images.append(p)
    return images


def create_split_dirs(output_root: Path, classes: List[str]) -> None:
    for split in ["train", "test"]:
        for cls in classes:
            (output_root / split / cls).mkdir(parents=True, exist_ok=True)


def reset_split_dirs(output_root: Path, classes: List[str]) -> None:
    """Remove old generated data to avoid mixed or stale labels."""
    for split in ["train", "test"]:
        split_path = output_root / split
        if split_path.exists():
            shutil.rmtree(split_path)
    create_split_dirs(output_root, classes)


def copy_with_unique_name(src: Path, dst_dir: Path, prefix: str) -> None:
    filename = f"{prefix}_{src.name}"
    target = dst_dir / filename

    if not target.exists():
        shutil.copy2(src, target)
        return

    stem = target.stem
    suffix = target.suffix
    counter = 1
    while True:
        candidate = dst_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return
        counter += 1


def balanced_sample(files: List[Path], max_count: int, seed: int) -> List[Path]:
    if len(files) <= max_count:
        return files

    # Deterministic downsampling prevents one class from dominating training.
    import random

    rng = random.Random(seed)
    shuffled = files.copy()
    rng.shuffle(shuffled)
    return shuffled[:max_count]


def bootstrap_dataset(
    dfu_root: Path,
    output_root: Path,
    test_size: float = 0.2,
    seed: int = SEED,
    max_per_class: int = 350,
) -> None:
    """
    Build 4-class progression dataset from deterministic source-folder mapping.

    Mapping used (no score heuristics):
    - normal: Patches/Normal(Healthy skin)
    - mild: Transfer-Learning images/internetSet + Transfer-Learning images/samples
    - moderate: Transfer-Learning images/Wound Images
    - severe: Patches/Abnormal(Ulcer) + Transfer-Learning images/Wound Images2

    This avoids random mixed classes from heuristic auto-labeling and is easier to audit.
    """
    classes = ["normal", "mild", "moderate", "severe"]
    reset_split_dirs(output_root, classes)

    class_sources: Dict[str, List[Path]] = {
        "normal": [
            dfu_root / "DFU" / "Patches" / "Normal(Healthy skin)",
        ],
        "mild": [
            dfu_root / "DFU" / "Transfer-Learning images" / "internetSet",
            dfu_root / "DFU" / "Transfer-Learning images" / "samples",
        ],
        "moderate": [
            dfu_root / "DFU" / "Transfer-Learning images" / "Wound Images",
        ],
        "severe": [
            dfu_root / "DFU" / "Patches" / "Abnormal(Ulcer)",
            dfu_root / "DFU" / "Transfer-Learning images" / "Wound Images2",
        ],
    }

    grouped: Dict[str, List[Path]] = {}
    print("Building source-mapped progression dataset:")
    for cls in classes:
        collected: List[Path] = []
        for source_folder in class_sources[cls]:
            files = collect_images(source_folder)
            print(f"  {cls} <- {source_folder} ({len(files)} images)")
            collected.extend(files)

        if len(collected) < 12:
            raise ValueError(
                f"Class '{cls}' has only {len(collected)} images. Need at least 12."
            )

        grouped[cls] = balanced_sample(collected, max_count=max_per_class, seed=seed)

    for cls in classes:
        collected = grouped[cls]
        train_files, test_files = train_test_split(
            collected,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )

        train_target = output_root / "train" / cls
        test_target = output_root / "test" / cls

        for src in train_files:
            copy_with_unique_name(src, train_target, cls)

        for src in test_files:
            copy_with_unique_name(src, test_target, cls)

        print(
            f"Class '{cls}': total={len(collected)}, train={len(train_files)}, test={len(test_files)}"
        )

    print(f"\nDone. Dataset ready at: {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap DFU progression dataset from current workspace")
    parser.add_argument(
        "--dfu_root",
        default=str(Path(__file__).resolve().parents[1] / "DFU_Dataset"),
        help="Path to DFU_Dataset root",
    )
    parser.add_argument(
        "--output_root",
        default=DATA_ROOT,
        help="Path to output dataset root",
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=350,
        help="Maximum images per class before train/test split",
    )
    args = parser.parse_args()

    dfu_root = Path(args.dfu_root)
    output_root = Path(args.output_root)

    if not dfu_root.exists():
        raise FileNotFoundError(f"DFU root not found: {dfu_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    bootstrap_dataset(
        dfu_root=dfu_root,
        output_root=output_root,
        test_size=args.test_size,
        max_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()
