from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _normalize_rows(confusion: np.ndarray) -> np.ndarray:
    matrix = np.asarray(confusion, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    out = np.zeros_like(matrix, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(matrix, row_sums, out=out, where=row_sums > 0)
    out[np.isnan(out)] = 0.0
    return out


def _draw_confusion(
    confusion: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
    normalize: bool,
    title: str,
) -> None:
    matrix = _normalize_rows(confusion) if normalize else np.asarray(confusion, dtype=np.float64)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    side = max(6.0, 1.0 + 0.8 * len(class_names))
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_idx = np.arange(len(class_names))
    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_title(title, fontsize=11)

    threshold = float(matrix.max() / 2.0) if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            if normalize:
                text = f"{value:.2f}"
            else:
                text = str(int(round(value)))
            color = "white" if value > threshold else "black"
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_set(
    confusion: np.ndarray,
    class_names: Sequence[str],
    out_dir: str | Path,
) -> Tuple[Path, Path, Path]:
    base_dir = Path(out_dir).resolve()
    counts_path = base_dir / "confusion_matrix_counts.png"
    norm_path = base_dir / "confusion_matrix_norm.png"
    default_path = base_dir / "confusion_matrix.png"

    _draw_confusion(
        confusion=confusion,
        class_names=class_names,
        out_path=counts_path,
        normalize=False,
        title="Confusion Matrix (Counts)",
    )
    _draw_confusion(
        confusion=confusion,
        class_names=class_names,
        out_path=norm_path,
        normalize=True,
        title="Confusion Matrix (Row-Normalized)",
    )
    _draw_confusion(
        confusion=confusion,
        class_names=class_names,
        out_path=default_path,
        normalize=False,
        title="Confusion Matrix",
    )

    return counts_path, norm_path, default_path

