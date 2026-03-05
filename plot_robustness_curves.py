import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate(rows: List[Dict[str, str]], metric: str) -> Dict[str, List[Tuple[float, float, float]]]:
    grouped: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    for row in rows:
        corr = str(row["corruption"])
        sev = float(row["severity"])
        grouped[(corr, sev)].append(float(row[metric]))

    out: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for (corr, sev), vals in grouped.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[corr].append((sev, float(arr.mean()), float(arr.std(ddof=0))))

    for corr in list(out.keys()):
        out[corr] = sorted(out[corr], key=lambda x: x[0])
    return out


def plot(rows: List[Dict[str, str]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    acc = aggregate(rows, metric="accuracy")
    f1 = aggregate(rows, metric="macro_f1")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for corr, curve in sorted(acc.items()):
        x = [c[0] for c in curve]
        y = [c[1] for c in curve]
        e = [c[2] for c in curve]
        axes[0].plot(x, y, marker="o", label=corr)
        axes[0].fill_between(x, np.array(y) - np.array(e), np.array(y) + np.array(e), alpha=0.15)

    for corr, curve in sorted(f1.items()):
        x = [c[0] for c in curve]
        y = [c[1] for c in curve]
        e = [c[2] for c in curve]
        axes[1].plot(x, y, marker="o", label=corr)
        axes[1].fill_between(x, np.array(y) - np.array(e), np.array(y) + np.array(e), alpha=0.15)

    axes[0].set_title("Robustness Accuracy")
    axes[0].set_xlabel("Severity")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Robustness Macro F1")
    axes[1].set_xlabel("Severity")
    axes[1].set_ylabel("Macro F1")
    axes[1].grid(alpha=0.3)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels) // 2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot robustness curves from robustness_results.csv")
    parser.add_argument("--input", type=str, required=True, help="Path to robustness_results.csv")
    parser.add_argument("--output", type=str, default="robustness_curves.png")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"input not found: {in_path}")

    rows = load_rows(in_path)
    if not rows:
        raise RuntimeError("empty robustness csv")

    plot(rows, Path(args.output))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
