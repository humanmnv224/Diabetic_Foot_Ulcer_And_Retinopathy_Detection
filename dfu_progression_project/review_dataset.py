import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}:
            yield p


CLASS_SCORE_RANGES: Dict[str, Tuple[float, float]] = {
    "mild": (0.00, 0.35),
    "moderate": (0.25, 0.60),
    "severe": (0.50, 1.00),
}


def estimate_severity_score(img_bgr: np.ndarray) -> float:
    """Estimate wound severity using simple color/coverage cues in [0, 1]."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    # Heuristic lesion mask: saturated/darker zones often correlate with tissue damage.
    lesion_mask = (s > 45) & (v < 205)
    lesion_ratio = float(np.mean(lesion_mask))

    # Red/inflamed tissue indicator.
    red_mask = (r > g + 20) & (r > b + 20) & (r > 80)
    redness_ratio = float(np.mean(red_mask))

    # Dark tissue indicator.
    dark_ratio = float(np.mean(v < 95))

    score = 0.45 * lesion_ratio + 0.35 * redness_ratio + 0.20 * dark_ratio
    return float(np.clip(score, 0.0, 1.0))


def is_suspect_for_class(score: float, cls: str) -> Tuple[bool, float]:
    low, high = CLASS_SCORE_RANGES[cls]
    if low <= score <= high:
        return False, 0.0
    if score < low:
        return True, low - score
    return True, score - high


def collect_review_items(class_dir: Path, cls: str, suspects_first: bool) -> List[Tuple[Path, float, bool, float]]:
    items: List[Tuple[Path, float, bool, float]] = []
    for img_path in iter_images(class_dir):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        score = estimate_severity_score(img)
        suspect, distance = is_suspect_for_class(score, cls)
        items.append((img_path, score, suspect, distance))

    if suspects_first:
        items.sort(key=lambda t: (not t[2], -t[3], -t[1]))
    return items


def propose_target_class(score: float) -> str:
    """Map score to class using conservative bins."""
    if score < 0.28:
        return "mild"
    if score < 0.52:
        return "moderate"
    return "severe"


def auto_relabel_split(dataset_root: Path, split: str, margin: float) -> None:
    """Auto-move only clearly out-of-range samples to reduce obvious class mixing."""
    valid_classes = ["mild", "moderate", "severe"]
    moved = 0
    print(f"Auto relabel on split='{split}' with margin={margin:.3f}")

    for cls in valid_classes:
        class_dir = dataset_root / split / cls
        if not class_dir.exists():
            continue

        items = collect_review_items(class_dir, cls, suspects_first=False)
        class_moved = 0

        for img_path, score, suspect, distance in items:
            if not suspect or distance < margin:
                continue

            target = propose_target_class(score)
            if target == cls:
                continue

            move_image(img_path, dataset_root, split, target)
            class_moved += 1
            moved += 1

        print(f"  {cls:8s} moved={class_moved:4d}")

    print(f"Total auto-moved in {split}: {moved}")


def move_image(src: Path, dataset_root: Path, split: str, target_class: str):
    target_dir = dataset_root / split / target_class
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / src.name

    if target_path.exists():
        base = target_path.stem
        ext = target_path.suffix
        idx = 1
        while True:
            candidate = target_dir / f"{base}_{idx}{ext}"
            if not candidate.exists():
                target_path = candidate
                break
            idx += 1

    src.replace(target_path)


def review_split(
    dataset_root: Path,
    split: str,
    suspects_first: bool,
    only_suspects: bool,
) -> None:
    valid_classes = ["mild", "moderate", "severe"]
    for cls in valid_classes:
        class_dir = dataset_root / split / cls
        if not class_dir.exists():
            continue

        items = collect_review_items(class_dir, cls, suspects_first=suspects_first)
        for img_path, score, suspect, _ in items:
            if only_suspects and not suspect:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            disp = img.copy()
            text = f"{split}/{cls} | 1:mild 2:moderate 3:severe s:skip q:quit"
            text2 = f"score={score:.3f} suspect={'yes' if suspect else 'no'}"
            cv2.putText(disp, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(disp, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Dataset Review", disp)
            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            if key == ord("s"):
                continue
            if key == ord("1"):
                target = "mild"
            elif key == ord("2"):
                target = "moderate"
            elif key == ord("3"):
                target = "severe"
            else:
                continue

            if target != cls:
                move_image(img_path, dataset_root, split, target)
                print(f"Moved: {img_path.name} -> {split}/{target}")

    cv2.destroyAllWindows()


def report_split(dataset_root: Path, split: str) -> None:
    valid_classes = ["mild", "moderate", "severe"]
    print(f"Review report for split='{split}'")
    for cls in valid_classes:
        class_dir = dataset_root / split / cls
        if not class_dir.exists():
            print(f"  {cls:8s} total=0 suspect=0")
            continue

        items = collect_review_items(class_dir, cls, suspects_first=False)
        total = len(items)
        suspect = sum(1 for _, _, is_suspect, _ in items if is_suspect)
        pct = (100.0 * suspect / total) if total else 0.0
        print(f"  {cls:8s} total={total:4d} suspect={suspect:4d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Manual dataset reviewer for mild/moderate/severe")
    parser.add_argument(
        "--dataset_root",
        default=str(Path(__file__).resolve().parent / "dataset"),
        help="Path to dataset root containing train/test",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Which split to review",
    )
    parser.add_argument(
        "--suspects_first",
        action="store_true",
        help="Show likely mislabeled images first",
    )
    parser.add_argument(
        "--only_suspects",
        action="store_true",
        help="Only review likely mislabeled images",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print suspected-mix report and exit",
    )
    parser.add_argument(
        "--auto_relabel",
        action="store_true",
        help="Automatically relabel clearly out-of-range mild/moderate/severe samples",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.10,
        help="Minimum out-of-range distance needed before auto move",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if args.report:
        report_split(dataset_root, args.split)
        return

    if args.auto_relabel:
        auto_relabel_split(dataset_root, args.split, margin=args.margin)
        return

    review_split(
        dataset_root,
        args.split,
        suspects_first=args.suspects_first,
        only_suspects=args.only_suspects,
    )


if __name__ == "__main__":
    main()
