#!/usr/bin/env python3
"""Evaluate FGAesQ on three FGAesthetics pair-label test files.

The model, pair preprocessing, forward pass, score aggregation, and confidence
adjustment are imported unchanged from the official ``inference_series.py``.
This file only adapts the six-column annotations to the official input objects
and computes the four metrics used by ``xiangsitu_yanzheng_final.py``.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scipy.stats import spearmanr
from sklearn.metrics import f1_score


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_NAME_PATTERN = re.compile(r"^(?P<series>\d+)-(?P<index>\d+)$")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "FGAesQ.pt"
DEFAULT_TEST_FILES = (
    Path("/workspace/wjj/FGAesthetics/aigc/aigc_test.txt"),
    Path("/workspace/wjj/FGAesthetics/crop/crop_test.txt"),
    Path("/workspace/wjj/FGAesthetics/natural/natural_test.txt"),
)


@dataclass(frozen=True)
class PairRecord:
    series_id: str
    img1_idx: int
    img2_idx: int
    probability: float
    rank1: int
    rank2: int


@dataclass(frozen=True)
class EvaluationMetrics:
    pair_accuracy: float
    pair_f1: float
    series_best_accuracy: float
    series_rank_srcc: float
    evaluated_pairs: int


def normalize_series_id(series_id: str) -> str:
    """Match the ID normalization in xiangsitu_yanzheng_final.py."""
    return str(int(series_id))


def parse_test_file(
    txt_path: Path,
) -> tuple[dict[str, dict[int, int]], list[PairRecord]]:
    """Parse the six-column FGAesthetics annotation format.

    The resulting series/image insertion order matches the reference parser,
    which matters only in the unlikely event of tied prediction scores.
    """
    raw_series_info: dict[str, dict[int, int]] = defaultdict(dict)
    raw_pairs: list[tuple[str, int, int, float, int, int]] = []

    with txt_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(
                    f"{txt_path}:{line_number}: expected 6 columns, got {len(parts)}"
                )

            try:
                series_id = parts[0]
                img1_idx = int(parts[1])
                img2_idx = int(parts[2])
                probability = float(parts[3])
                rank1 = int(parts[4])
                rank2 = int(parts[5])
                normalize_series_id(series_id)
            except ValueError as exc:
                raise ValueError(
                    f"{txt_path}:{line_number}: invalid annotation: {line.strip()}"
                ) from exc

            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"{txt_path}:{line_number}: probability must be in [0, 1]"
                )

            raw_series_info[series_id][img1_idx] = rank1
            raw_series_info[series_id][img2_idx] = rank2
            raw_pairs.append(
                (series_id, img1_idx, img2_idx, probability, rank1, rank2)
            )

    if not raw_pairs:
        raise ValueError(f"No valid image pairs found in {txt_path}")

    # The reference validator normalizes IDs only after parsing.
    series_info: dict[str, dict[int, int]] = {}
    for raw_series_id, image_ranks in raw_series_info.items():
        series_info[normalize_series_id(raw_series_id)] = image_ranks

    pair_records = [
        PairRecord(
            series_id=normalize_series_id(series_id),
            img1_idx=img1_idx,
            img2_idx=img2_idx,
            probability=probability,
            rank1=rank1,
            rank2=rank2,
        )
        for series_id, img1_idx, img2_idx, probability, rank1, rank2 in raw_pairs
    ]
    return series_info, pair_records


def index_image_directory(image_dir: Path) -> dict[tuple[str, int], Path]:
    """Index flat ``{series_id}-{image_index}.ext`` image files."""
    image_index: dict[tuple[str, int], Path] = {}

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue

        match = IMAGE_NAME_PATTERN.fullmatch(image_path.stem)
        if match is None:
            continue

        key = (
            normalize_series_id(match.group("series")),
            int(match.group("index")),
        )
        previous = image_index.get(key)
        if previous is not None:
            raise ValueError(
                f"Duplicate image key {key}: {previous.name} and {image_path.name}"
            )
        image_index[key] = image_path

    return image_index


def build_official_inputs(
    series_info: Mapping[str, Mapping[int, int]],
    pair_records: Sequence[PairRecord],
    image_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Adapt the txt-defined test pairs to official inference structures."""
    indexed_images = index_image_directory(image_dir)
    required_keys = {
        (series_id, image_idx)
        for series_id, image_ranks in series_info.items()
        for image_idx in image_ranks
    }
    missing_keys = sorted(required_keys - indexed_images.keys())
    if missing_keys:
        examples = ", ".join(f"{sid}-{idx:02d}" for sid, idx in missing_keys[:10])
        suffix = " ..." if len(missing_keys) > 10 else ""
        raise FileNotFoundError(
            f"Missing {len(missing_keys)} referenced images in {image_dir}: "
            f"{examples}{suffix}"
        )

    series_data: dict[str, dict[str, Any]] = {}
    for series_id, image_ranks in series_info.items():
        image_mapping = {
            image_idx: indexed_images[(series_id, image_idx)].name
            for image_idx in image_ranks
        }
        series_data[series_id] = {
            "folder": str(image_dir),
            "images": [image_mapping[idx] for idx in sorted(image_mapping)],
            "image_mapping": image_mapping,
        }

    official_pairs = [
        {
            "series_id": record.series_id,
            "img1_idx": record.img1_idx,
            "img2_idx": record.img2_idx,
            "img1_name": indexed_images[(record.series_id, record.img1_idx)].name,
            "img2_name": indexed_images[(record.series_id, record.img2_idx)].name,
        }
        for record in pair_records
    ]
    return series_data, official_pairs


def quantize_official_scores(
    image_scores: Mapping[tuple[str, int], float],
) -> dict[tuple[str, int], float]:
    """Reproduce save_series_result's four-decimal output/readback step."""
    return {key: float(f"{float(score):.4f}") for key, score in image_scores.items()}


def evaluate_pair_metrics(
    pair_records: Sequence[PairRecord],
    image_scores: Mapping[tuple[str, int], float],
) -> tuple[float, float, int]:
    """Match evaluate_pair_accuracy_from_results in the reference validator."""
    correct_predictions = 0
    labels: list[int] = []
    predictions: list[int] = []

    for record in pair_records:
        if record.probability == 0.5:
            continue

        score1 = image_scores.get((record.series_id, record.img1_idx))
        score2 = image_scores.get((record.series_id, record.img2_idx))
        if score1 is None or score2 is None:
            continue

        is_correct = (
            record.probability > 0.5 and score1 > score2
        ) or (
            record.probability < 0.5 and score1 < score2
        )
        correct_predictions += int(is_correct)
        labels.append(1 if record.probability > 0.5 else 0)
        predictions.append(1 if score1 > score2 else 0)

    total_pairs = len(labels)
    accuracy = correct_predictions / total_pairs if total_pairs else 0.0
    macro_f1 = (
        float(f1_score(labels, predictions, average="macro"))
        if labels
        else 0.0
    )
    return accuracy, macro_f1, total_pairs


def evaluate_series_best_accuracy(
    series_info: Mapping[str, Mapping[int, int]],
    image_scores: Mapping[tuple[str, int], float],
) -> float:
    """Compute the reference length-weighted series-best accuracy."""
    total_correct = 0
    total_weight = 0

    for series_id, image_ranks in series_info.items():
        series_scores = {
            image_idx: image_scores[(series_id, image_idx)]
            for image_idx in image_ranks
            if (series_id, image_idx) in image_scores
        }
        if not series_scores:
            continue

        best_predicted_idx = max(series_scores, key=series_scores.get)
        best_true_idx = next(
            (image_idx for image_idx, rank in image_ranks.items() if rank == 1),
            None,
        )
        if best_true_idx is None:
            continue

        weight = len(image_ranks)
        total_correct += int(best_predicted_idx == best_true_idx) * weight
        total_weight += weight

    return total_correct / total_weight if total_weight else 0.0


def evaluate_series_rank_correlation(
    series_info: Mapping[str, Mapping[int, int]],
    image_scores: Mapping[tuple[str, int], float],
) -> float:
    """Compute the reference length-weighted per-series Spearman correlation."""
    correlations: list[float] = []
    weights: list[int] = []

    for series_id, image_ranks in series_info.items():
        if len(image_ranks) < 2:
            continue

        predicted_scores: list[float] = []
        negative_true_ranks: list[int] = []
        for image_idx, true_rank in image_ranks.items():
            score = image_scores.get((series_id, image_idx))
            if score is not None:
                predicted_scores.append(score)
                negative_true_ranks.append(-true_rank)

        if len(predicted_scores) < 2:
            continue

        try:
            correlation, _ = spearmanr(predicted_scores, negative_true_ranks)
            correlation = float(correlation)
        except Exception:
            continue

        if not math.isnan(correlation):
            correlations.append(correlation)
            weights.append(len(predicted_scores))

    total_weight = sum(weights)
    if not correlations or not total_weight:
        return 0.0
    return sum(value * weight for value, weight in zip(correlations, weights)) / total_weight


def calculate_metrics(
    series_info: Mapping[str, Mapping[int, int]],
    pair_records: Sequence[PairRecord],
    image_scores: Mapping[tuple[str, int], float],
) -> EvaluationMetrics:
    pair_accuracy, pair_f1, evaluated_pairs = evaluate_pair_metrics(
        pair_records, image_scores
    )
    return EvaluationMetrics(
        pair_accuracy=pair_accuracy,
        pair_f1=pair_f1,
        series_best_accuracy=evaluate_series_best_accuracy(
            series_info, image_scores
        ),
        series_rank_srcc=evaluate_series_rank_correlation(
            series_info, image_scores
        ),
        evaluated_pairs=evaluated_pairs,
    )


def print_metrics(dataset_name: str, metrics: EvaluationMetrics) -> None:
    print(dataset_name)
    print(
        "Pair Accuracy:       "
        f"{metrics.pair_accuracy:.4f} ({metrics.pair_accuracy * 100:.2f}%)"
    )
    print(f"Pair F1:             {metrics.pair_f1:.4f}")
    print(
        "Series Best (s-Acc): "
        f"{metrics.series_best_accuracy:.4f} "
        f"({metrics.series_best_accuracy * 100:.2f}%)"
    )
    print(f"Series SRCC:         {metrics.series_rank_srcc:.4f}", flush=True)


def infer_dataset_name(txt_path: Path) -> str:
    name = txt_path.stem
    return name[:-5] if name.endswith("_test") else name


def evaluate_dataset(
    *,
    model: Any,
    official_inference: Any,
    txt_path: Path,
    device: Any,
    batch_size: int,
    num_workers: int,
    position: int,
    total: int,
) -> EvaluationMetrics:
    image_dir = txt_path.parent / "img"
    if not txt_path.is_file():
        raise FileNotFoundError(f"Test annotation not found: {txt_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    dataset_name = infer_dataset_name(txt_path)

    series_info, pair_records = parse_test_file(txt_path)
    series_data, official_pairs = build_official_inputs(
        series_info, pair_records, image_dir
    )
    image_count = sum(len(images) for images in series_info.values())

    # These are the unmodified official series-mode inference functions.
    image_scores_all = official_inference.evaluate_pairs(
        model=model,
        pairs=official_pairs,
        series_data=series_data,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    adjusted_scores = official_inference.apply_confidence_adjustment(
        image_scores_all, series_data
    )

    if len(adjusted_scores) != image_count:
        raise RuntimeError(
            f"Inference returned {len(adjusted_scores)} of {image_count} image scores"
        )

    # The reference validator reads official result files whose scores are
    # written with ``:.4f``. Quantizing here reproduces that exact behavior
    # without requiring thousands of intermediate result files.
    evaluation_scores = quantize_official_scores(adjusted_scores)
    metrics = calculate_metrics(series_info, pair_records, evaluation_scores)
    print_metrics(dataset_name, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run official FGAesQ series inference and the four standard "
            "FGAesthetics metrics on three test txt files."
        )
    )
    parser.add_argument(
        "test_files",
        nargs="*",
        type=Path,
        metavar="TEST_TXT",
        help=(
            "Optional: three *_test.txt files; omitted values use the default "
            "paths configured in this script."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"FGAesQ checkpoint (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0)")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Inference batch size (default: 64)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers (default: 8; use 0 for low-memory troubleshooting)",
    )
    args = parser.parse_args()

    if not args.test_files:
        args.test_files = list(DEFAULT_TEST_FILES)
    elif len(args.test_files) != 3:
        parser.error("provide exactly three test txt files, or omit them to use defaults")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    return args


def resolve_device(requested_device: str) -> Any:
    import torch

    device = torch.device(requested_device)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        return torch.device("cpu")

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    if device_index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {device_index} does not exist; "
            f"available device count is {torch.cuda.device_count()}"
        )
    return device


def main() -> dict[str, EvaluationMetrics]:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    test_files = [path.expanduser().resolve() for path in args.test_files]

    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Importing this module is intentional: all model and inference behavior
    # stays owned by the official implementation.
    import inference_series as official_inference

    device = resolve_device(args.device)
    model = official_inference.load_model(str(model_path), device)

    results: dict[str, EvaluationMetrics] = {}
    for position, txt_path in enumerate(test_files, start=1):
        name = infer_dataset_name(txt_path)
        if name in results:
            raise ValueError(f"Duplicate dataset name inferred from input files: {name}")
        results[name] = evaluate_dataset(
            model=model,
            official_inference=official_inference,
            txt_path=txt_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            position=position,
            total=len(test_files),
        )

    return results


if __name__ == "__main__":
    main()
