import csv
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

from ARIL.aril import aril
from HAR.har import har1, har3
from SignFi.signfi import signfi
from StanFi.stanfi import stanfi
from augmentation import apply_train_augmentation
from DataLoader.tensordata import TensorData
from protocol_utils import labels_to_index


@dataclass
class SampleRecord:
    index: int
    source_split: str
    source_index: int
    label_index: int
    sample_id: str
    subject: Optional[str] = None
    session: Optional[str] = None
    environment: Optional[str] = None


@dataclass
class DatasetBundle:
    dataset_name: str
    x_all: np.ndarray
    y_all: np.ndarray
    samples: List[SampleRecord]
    num_classes: int
    input_size: int
    class_names: List[str]


@dataclass
class SplitIndices:
    train: List[int]
    val: List[int]
    test: List[int]
    mode_requested: str
    mode_used: str
    group_key_requested: str
    group_key_used: Optional[str]
    warnings: List[str]


DATASET_LOADERS = {
    "aril": aril,
    "har-1": har1,
    "har-3": har3,
    "signfi": signfi,
    "stanfi": stanfi,
}


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False)


def _ensure_one_hot(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        idx = y.astype(np.int64)
        classes = int(idx.max() + 1) if num_classes is None else int(num_classes)
        out = np.zeros((idx.shape[0], classes), dtype=np.float32)
        out[np.arange(idx.shape[0]), idx] = 1.0
        return out
    if y.ndim == 2 and y.shape[1] == 1:
        idx = y.reshape(-1).astype(np.int64)
        classes = int(idx.max() + 1) if num_classes is None else int(num_classes)
        out = np.zeros((idx.shape[0], classes), dtype=np.float32)
        out[np.arange(idx.shape[0]), idx] = 1.0
        return out
    return y.astype(np.float32)


def _load_metadata(metadata_path: Optional[str]) -> Dict[Tuple[str, int], Dict[str, Optional[str]]]:
    if not metadata_path:
        return {}

    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("rows", [])
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError("metadata json must be list or dict with key 'rows'")
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    mapping: Dict[Tuple[str, int], Dict[str, Optional[str]]] = {}
    for row in rows:
        split = str(row.get("source_split", "")).strip().lower()
        idx_raw = row.get("source_index", row.get("index", None))
        if idx_raw is None or not split:
            continue
        src_idx = int(idx_raw)
        mapping[(split, src_idx)] = {
            "subject": _norm_meta(row.get("subject")),
            "session": _norm_meta(row.get("session")),
            "environment": _norm_meta(row.get("environment")),
        }
    return mapping


def _norm_meta(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def load_dataset_bundle(dataset_name: str, metadata_path: Optional[str] = None) -> DatasetBundle:
    key = dataset_name.lower()
    if key not in DATASET_LOADERS:
        raise ValueError(f"unsupported dataset: {dataset_name}")

    loader = DATASET_LOADERS[key]
    cwd = Path.cwd()
    try:
        x_train, y_train, x_test, y_test = loader()
    finally:
        os.chdir(cwd)

    x_train_np = _ensure_float32(_to_numpy(x_train))
    x_test_np = _ensure_float32(_to_numpy(x_test))
    y_train_np = _to_numpy(y_train)
    y_test_np = _to_numpy(y_test)

    y_train_oh = _ensure_one_hot(y_train_np)
    y_test_oh = _ensure_one_hot(y_test_np, num_classes=y_train_oh.shape[1])

    num_classes = int(max(y_train_oh.shape[1], y_test_oh.shape[1]))
    if y_train_oh.shape[1] != num_classes:
        y_train_oh = _ensure_one_hot(labels_to_index(y_train_oh), num_classes=num_classes)
    if y_test_oh.shape[1] != num_classes:
        y_test_oh = _ensure_one_hot(labels_to_index(y_test_oh), num_classes=num_classes)

    x_all = np.concatenate([x_train_np, x_test_np], axis=0)
    y_all = np.concatenate([y_train_oh.astype(np.float32), y_test_oh.astype(np.float32)], axis=0)
    label_idx = labels_to_index(y_all)

    metadata = _load_metadata(metadata_path)

    samples: List[SampleRecord] = []
    n_train = x_train_np.shape[0]
    for i in range(x_all.shape[0]):
        source_split = "train" if i < n_train else "test"
        source_index = i if i < n_train else i - n_train
        meta = metadata.get((source_split, source_index), {})
        sample_id = f"{source_split}_{source_index}"
        samples.append(
            SampleRecord(
                index=i,
                source_split=source_split,
                source_index=source_index,
                label_index=int(label_idx[i]),
                sample_id=sample_id,
                subject=meta.get("subject"),
                session=meta.get("session"),
                environment=meta.get("environment"),
            )
        )

    input_size = int(x_all.shape[-1])
    class_names = [str(i) for i in range(num_classes)]
    return DatasetBundle(
        dataset_name=key,
        x_all=x_all,
        y_all=y_all,
        samples=samples,
        num_classes=num_classes,
        input_size=input_size,
        class_names=class_names,
    )


def _pick_group_key(samples: Sequence[SampleRecord], requested: str) -> Optional[str]:
    keys = ["subject", "session", "environment"]

    def coverage(name: str) -> Tuple[int, int]:
        values = [getattr(s, name) for s in samples]
        present = [v for v in values if v is not None]
        return len(present), len(set(present))

    if requested in keys:
        n_present, n_unique = coverage(requested)
        if n_present == len(samples) and n_unique > 1:
            return requested
        return None

    if requested == "auto":
        ranked: List[Tuple[int, str]] = []
        for key in keys:
            n_present, n_unique = coverage(key)
            if n_present == len(samples) and n_unique > 1:
                ranked.append((n_unique, key))
        ranked.sort(reverse=True)
        return ranked[0][1] if ranked else None

    return None


def _random_split(
    indices: np.ndarray,
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratified: bool,
) -> Tuple[List[int], List[int], List[int]]:
    if not stratified:
        rng = np.random.RandomState(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, min(int(round(n * test_ratio)), n - 1))
        n_val = max(0, min(int(round(n * val_ratio)), n - n_test - 1))
        test_idx = shuffled[:n_test]
        val_idx = shuffled[n_test:n_test + n_val]
        train_idx = shuffled[n_test + n_val:]
        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    try:
        split_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        trainval_idx, test_idx = next(split_test.split(indices, labels))
    except Exception:
        return _random_split(
            indices=indices,
            labels=labels,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            stratified=False,
        )

    remaining_indices = indices[trainval_idx]
    remaining_labels = labels[trainval_idx]
    val_share = val_ratio / max(train_ratio + val_ratio, 1e-12)

    if len(np.unique(remaining_labels)) < 2 or len(remaining_indices) < 3:
        rng = np.random.RandomState(seed)
        shuffled = remaining_indices.copy()
        rng.shuffle(shuffled)
        n_val = max(1, min(int(round(len(shuffled) * val_share)), len(shuffled) - 1))
        val_idx = shuffled[:n_val]
        train_idx = shuffled[n_val:]
    else:
        try:
            split_val = StratifiedShuffleSplit(n_splits=1, test_size=val_share, random_state=seed)
            train_sub, val_sub = next(split_val.split(remaining_indices, remaining_labels))
            train_idx = remaining_indices[train_sub]
            val_idx = remaining_indices[val_sub]
        except Exception:
            rng = np.random.RandomState(seed)
            shuffled = remaining_indices.copy()
            rng.shuffle(shuffled)
            n_val = max(1, min(int(round(len(shuffled) * val_share)), len(shuffled) - 1))
            val_idx = shuffled[:n_val]
            train_idx = shuffled[n_val:]

    return train_idx.tolist(), val_idx.tolist(), indices[test_idx].tolist()


def _group_split(
    indices: np.ndarray,
    groups: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    unique_groups = np.unique(groups)
    if unique_groups.shape[0] < 3:
        raise ValueError("not enough unique groups for train/val/test")

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_sub, test_sub = next(gss_test.split(indices, groups=groups))
    trainval_idx = indices[trainval_sub]
    test_idx = indices[test_sub]

    groups_tv = groups[trainval_sub]
    val_share = val_ratio / max(1e-12, (1.0 - test_ratio))
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_share, random_state=seed)
    train_sub, val_sub = next(gss_val.split(trainval_idx, groups=groups_tv))
    train_idx = trainval_idx[train_sub]
    val_idx = trainval_idx[val_sub]

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def _label_counts(indices: Sequence[int], labels: np.ndarray) -> Dict[int, int]:
    out: Dict[int, int] = defaultdict(int)
    for idx in indices:
        out[int(labels[idx])] += 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def generate_or_load_split(
    bundle: DatasetBundle,
    split_file: Path,
    split_mode: str,
    group_key: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    stratified_random: bool,
    reuse_existing: bool,
) -> Tuple[SplitIndices, Dict[str, Any]]:
    if reuse_existing and split_file.exists():
        payload = json.loads(split_file.read_text(encoding="utf-8"))
        split = SplitIndices(
            train=[int(v) for v in payload["train"]],
            val=[int(v) for v in payload["val"]],
            test=[int(v) for v in payload["test"]],
            mode_requested=str(payload.get("mode_requested", split_mode)),
            mode_used=str(payload.get("mode_used", split_mode)),
            group_key_requested=str(payload.get("group_key_requested", group_key)),
            group_key_used=payload.get("group_key_used"),
            warnings=list(payload.get("warnings", [])),
        )
        return split, {"split_file": str(split_file), "loaded_existing_split": True, "split_summary": payload.get("split_summary", {})}

    labels = np.asarray([s.label_index for s in bundle.samples], dtype=np.int64)
    indices = np.arange(len(bundle.samples), dtype=np.int64)

    warnings: List[str] = []
    mode_used = split_mode
    used_group: Optional[str] = None

    if split_mode == "predefined":
        train_idx = [s.index for s in bundle.samples if s.source_split == "train"]
        test_idx = [s.index for s in bundle.samples if s.source_split == "test"]

        if not train_idx or not test_idx:
            warnings.append("predefined split unavailable; fallback=random")
            mode_used = "random"
            train_idx, val_idx, test_idx = _random_split(
                indices=indices,
                labels=labels,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                stratified=stratified_random,
            )
        else:
            train_idx_np = np.asarray(train_idx, dtype=np.int64)
            y_train = labels[train_idx_np]
            if len(np.unique(y_train)) >= 2 and len(train_idx_np) >= 3 and val_ratio > 0:
                val_share = val_ratio / max(train_ratio + val_ratio, 1e-12)
                try:
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_share, random_state=seed)
                    tr_sub, va_sub = next(sss.split(train_idx_np, y_train))
                    val_idx = train_idx_np[va_sub].tolist()
                    train_idx = train_idx_np[tr_sub].tolist()
                except Exception:
                    rng = np.random.RandomState(seed)
                    shuffled = train_idx_np.copy()
                    rng.shuffle(shuffled)
                    n_val = max(1, min(int(round(len(shuffled) * val_share)), len(shuffled) - 1))
                    val_idx = shuffled[:n_val].tolist()
                    train_idx = shuffled[n_val:].tolist()
            else:
                val_idx = []

    elif split_mode == "group":
        used_group = _pick_group_key(bundle.samples, group_key)
        if used_group is None:
            warnings.append(
                "group metadata unavailable; fallback=random (TODO: provide metadata CSV/JSON with source_split/source_index and subject/session/environment)"
            )
            mode_used = "random"
            train_idx, val_idx, test_idx = _random_split(
                indices=indices,
                labels=labels,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                stratified=stratified_random,
            )
        else:
            groups = np.asarray([getattr(s, used_group) for s in bundle.samples])
            train_idx, val_idx, test_idx = _group_split(
                indices=indices,
                groups=groups,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
    else:
        train_idx, val_idx, test_idx = _random_split(
            indices=indices,
            labels=labels,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            stratified=stratified_random,
        )

    split = SplitIndices(
        train=sorted(train_idx),
        val=sorted(val_idx),
        test=sorted(test_idx),
        mode_requested=split_mode,
        mode_used=mode_used,
        group_key_requested=group_key,
        group_key_used=used_group,
        warnings=warnings,
    )

    split_summary = {
        "counts": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
            "total": len(bundle.samples),
        },
        "label_counts": {
            "train": _label_counts(split.train, labels),
            "val": _label_counts(split.val, labels),
            "test": _label_counts(split.test, labels),
        },
    }

    payload = {
        "dataset": bundle.dataset_name,
        "seed": int(seed),
        "mode_requested": split.mode_requested,
        "mode_used": split.mode_used,
        "group_key_requested": split.group_key_requested,
        "group_key_used": split.group_key_used,
        "warnings": split.warnings,
        "train": split.train,
        "val": split.val,
        "test": split.test,
        "split_summary": split_summary,
        "ratios": {
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio),
        },
        "stratified_random": bool(stratified_random),
    }
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return split, {"split_file": str(split_file), "loaded_existing_split": False, "split_summary": split_summary}


def _quick_hash_array(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    digest = hashlib.sha1()
    digest.update(str(x.shape).encode("utf-8"))
    digest.update(str(x.dtype).encode("utf-8"))
    raw = x.view(np.uint8)
    if raw.size <= 8192:
        digest.update(raw.tobytes())
    else:
        digest.update(raw[:4096].tobytes())
        digest.update(raw[-4096:].tobytes())
    return digest.hexdigest()


def leakage_checks(bundle: DatasetBundle, split: SplitIndices, hash_mode: str = "quick") -> Dict[str, Any]:
    train_ids = {bundle.samples[i].sample_id for i in split.train}
    val_ids = {bundle.samples[i].sample_id for i in split.val}
    test_ids = {bundle.samples[i].sample_id for i in split.test}

    report: Dict[str, Any] = {
        "id_overlap": {
            "train_val": len(train_ids.intersection(val_ids)),
            "train_test": len(train_ids.intersection(test_ids)),
            "val_test": len(val_ids.intersection(test_ids)),
        },
        "hash_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
        "group_overlap": {},
        "warnings": [],
    }

    if hash_mode == "quick":
        train_hash = {_quick_hash_array(bundle.x_all[i]) for i in split.train}
        val_hash = {_quick_hash_array(bundle.x_all[i]) for i in split.val}
        test_hash = {_quick_hash_array(bundle.x_all[i]) for i in split.test}
        report["hash_overlap"] = {
            "train_val": len(train_hash.intersection(val_hash)),
            "train_test": len(train_hash.intersection(test_hash)),
            "val_test": len(val_hash.intersection(test_hash)),
        }

    if split.group_key_used is not None:
        key = split.group_key_used
        train_groups = {getattr(bundle.samples[i], key) for i in split.train}
        val_groups = {getattr(bundle.samples[i], key) for i in split.val}
        test_groups = {getattr(bundle.samples[i], key) for i in split.test}
        train_groups.discard(None)
        val_groups.discard(None)
        test_groups.discard(None)
        report["group_overlap"] = {
            "group_key": key,
            "train_val": len(train_groups.intersection(val_groups)),
            "train_test": len(train_groups.intersection(test_groups)),
            "val_test": len(val_groups.intersection(test_groups)),
        }

    if any(report["id_overlap"].values()):
        report["warnings"].append("sample ID overlap detected")
    if any(report["hash_overlap"].values()):
        report["warnings"].append("sample hash overlap detected")
    if report["group_overlap"] and any(
        int(v) > 0 for k, v in report["group_overlap"].items() if k != "group_key"
    ):
        report["warnings"].append("group overlap detected")

    return report


def _build_tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorData:
    return TensorData(torch.FloatTensor(x), torch.FloatTensor(y))


def build_dataloaders(
    bundle: DatasetBundle,
    split: SplitIndices,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    augment_train: bool,
    augment_gaussian: bool = True,
    augment_paper_gaussian: bool = False,
    augment_shift: bool = True,
    shift_steps: int = 10,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    x_train = bundle.x_all[split.train]
    y_train = bundle.y_all[split.train]

    x_train_use, y_train_use = apply_train_augmentation(
        x_train=x_train,
        y_train=y_train,
        augment_train=bool(augment_train),
        augment_gaussian=bool(augment_gaussian),
        augment_paper_gaussian=bool(augment_paper_gaussian),
        augment_shift=bool(augment_shift),
        shift_steps=int(shift_steps),
        seed=int(seed),
    )

    x_val = bundle.x_all[split.val]
    y_val = bundle.y_all[split.val]
    x_test = bundle.x_all[split.test]
    y_test = bundle.y_all[split.test]

    train_ds = _build_tensor_dataset(_ensure_float32(x_train_use), _ensure_float32(y_train_use))
    val_ds = _build_tensor_dataset(_ensure_float32(x_val), _ensure_float32(y_val))
    test_ds = _build_tensor_dataset(_ensure_float32(x_test), _ensure_float32(y_test))

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    loader_kwargs: Dict[str, Any] = {"num_workers": int(num_workers)}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_batch_size),
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(eval_batch_size),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(eval_batch_size),
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader
