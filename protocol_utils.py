import csv
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_int_list(text: str) -> List[int]:
    values = []
    for raw in str(text).split(","):
        item = raw.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("empty integer list")
    return values


def parse_float_list(text: str) -> List[float]:
    values = []
    for raw in str(text).split(","):
        item = raw.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("empty float list")
    return values


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def get_git_commit_hash(repo_root: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def collect_environment_info(repo_root: Path) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "git_commit": get_git_commit_hash(repo_root),
        "repo_root": str(repo_root),
    }


def labels_to_index(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(np.int64)
    if y.ndim == 2 and y.shape[1] == 1:
        return y.reshape(-1).astype(np.int64)
    return np.argmax(y, axis=1).astype(np.int64)


def confusion_matrix_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def per_class_f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    denom = 2.0 * tp + fp + fn
    out = np.zeros_like(tp, dtype=np.float64)
    valid = denom > 0
    out[valid] = 2.0 * tp[valid] / denom[valid]
    return out


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    values = per_class_f1_from_confusion(cm)
    valid = cm.sum(axis=1) > 0
    if not np.any(valid):
        return 0.0
    return float(values[valid].mean())


def normalize_confusion_rows(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float64)
    sums = cm.sum(axis=1, keepdims=True)
    out = np.zeros_like(cm, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(cm, sums, out=out, where=sums > 0)
    out[np.isnan(out)] = 0.0
    return out


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, class_names: Sequence[str]) -> Dict[str, Any]:
    cm = confusion_matrix_from_predictions(y_true, y_pred, num_classes=num_classes)
    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0
    per_class_f1 = per_class_f1_from_confusion(cm)
    macro_f1 = macro_f1_from_confusion(cm)

    rows = []
    for idx in range(num_classes):
        rows.append(
            {
                "class_index": int(idx),
                "class_name": class_names[idx] if idx < len(class_names) else str(idx),
                "f1": float(per_class_f1[idx]),
                "support": int(cm[idx].sum()),
            }
        )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_rows": rows,
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": cm,
        "num_samples": total,
    }


def maybe_save_confusion_png(
    confusion: np.ndarray,
    class_names: Sequence[str],
    path: Path,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    matrix = normalize_confusion_rows(confusion) if normalize else confusion
    ensure_dir(path.parent)

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def measure_inference_latency(
    model: torch.nn.Module,
    sample_batch: torch.Tensor,
    device: torch.device,
    warmup: int = 10,
    iters: int = 30,
) -> Dict[str, float]:
    model.eval()
    batch = sample_batch.to(device)

    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(max(1, int(iters))):
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    ms_per_batch = float(1000.0 * elapsed / max(1, int(iters)))
    throughput = float(batch.shape[0] / (elapsed / max(1, int(iters)))) if elapsed > 0 else 0.0
    return {
        "latency_ms_per_batch": ms_per_batch,
        "throughput_samples_per_sec": throughput,
        "latency_warmup_iters": int(warmup),
        "latency_measure_iters": int(iters),
        "latency_batch_size": int(batch.shape[0]),
    }


def aggregate_mean_std(rows: Sequence[Dict[str, Any]], keys: Iterable[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(values.mean()) if len(values) else 0.0,
            "std": float(values.std(ddof=0)) if len(values) else 0.0,
        }
    return out
