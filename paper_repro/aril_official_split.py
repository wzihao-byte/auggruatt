from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler


def _resolve_aril_dir(aril_dir: str | Path | None) -> Path:
    if aril_dir is not None:
        return Path(aril_dir).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "ARIL").resolve()


def _to_one_hot(labels: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    idx = np.asarray(labels).reshape(-1).astype(np.int64)
    if idx.size == 0:
        raise ValueError("empty labels")
    if np.min(idx) < 0:
        raise ValueError("labels must be non-negative")

    classes = int(np.max(idx) + 1) if num_classes is None else int(num_classes)
    out = np.zeros((idx.shape[0], classes), dtype=np.float32)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


def _load_mat_payload(path: Path, data_key: str, label_key: str) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"missing ARIL official split file: {path}\n"
            "Expected files: train_data_split_amp.mat and test_data_split_amp.mat."
        )

    payload = sio.loadmat(str(path))
    if data_key not in payload or label_key not in payload:
        keys = ", ".join(sorted(k for k in payload.keys() if not k.startswith("__")))
        raise KeyError(
            f"{path.name} must include keys '{data_key}' and '{label_key}'. "
            f"Available keys: {keys}"
        )

    data = np.asarray(payload[data_key], dtype=np.float32)
    labels = np.asarray(payload[label_key]).reshape(-1).astype(np.int64)
    return data, labels


def load_aril_official_split(
    aril_dir: str | Path | None = None,
    task: str = "activity",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load ARIL official train/test split from dataset-provided .mat files.

    Returns:
        X_train: [N_train, T, D]
        y_train: one-hot [N_train, C]
        X_test:  [N_test, T, D]
        y_test:  one-hot [N_test, C]
        class_names: ['0', '1', ...]
    """
    aril_root = _resolve_aril_dir(aril_dir)

    task_key = str(task).strip().lower()
    if task_key == "activity":
        train_label_key = "train_activity_label"
        test_label_key = "test_activity_label"
    elif task_key == "location":
        train_label_key = "train_location_label"
        test_label_key = "test_location_label"
    else:
        raise ValueError("task must be 'activity' or 'location'")

    train_path = aril_root / "train_data_split_amp.mat"
    test_path = aril_root / "test_data_split_amp.mat"

    train_data_raw, train_labels = _load_mat_payload(
        path=train_path,
        data_key="train_data",
        label_key=train_label_key,
    )
    test_data_raw, test_labels = _load_mat_payload(
        path=test_path,
        data_key="test_data",
        label_key=test_label_key,
    )

    # ARIL .mat stores [N, D, T] and the model expects [N, T, D].
    train_data = np.transpose(train_data_raw, (0, 2, 1))
    test_data = np.transpose(test_data_raw, (0, 2, 1))

    # Match existing preprocessing: fit scaler on train only, then apply to both.
    train_2d = train_data.reshape(train_data.shape[0], -1)
    test_2d = test_data.reshape(test_data.shape[0], -1)

    scaler = StandardScaler().fit(train_2d)
    train_2d = scaler.transform(train_2d)
    test_2d = scaler.transform(test_2d)

    x_train = train_2d.reshape(train_data.shape).astype(np.float32)
    x_test = test_2d.reshape(test_data.shape).astype(np.float32)

    y_train = _to_one_hot(train_labels)
    y_test = _to_one_hot(test_labels, num_classes=y_train.shape[1])
    class_names = [str(i) for i in range(y_train.shape[1])]

    return x_train, y_train, x_test, y_test, class_names

