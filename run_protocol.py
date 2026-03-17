import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from dual_domain_model import MODEL_VARIANTS, TEMPORAL_POOLING_CHOICES, build_model, model_metadata
from protocol_corruptions import SUPPORTED_CORRUPTIONS, apply_corruption, resolve_severity_levels
from protocol_data import build_dataloaders, generate_or_load_split, leakage_checks, load_dataset_bundle
from protocol_utils import (
    aggregate_mean_std,
    classification_metrics,
    collect_environment_info,
    count_parameters,
    ensure_dir,
    labels_to_index,
    maybe_save_confusion_png,
    measure_inference_latency,
    normalize_confusion_rows,
    parse_float_list,
    parse_int_list,
    set_global_seed,
    timestamp_now,
    to_serializable,
    write_csv,
    write_json,
)
from results_layout import (
    build_experiment_layout,
    build_run_manifest,
    initialize_layout,
    relative_to_results_root,
    write_manifest_and_registry,
    write_results_readme,
    write_run_readme,
)
from tools.mixup import mixup


DATASET_CHOICES = ["aril", "har-1", "har-3", "signfi", "stanfi"]


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {path}")
    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("yaml config requires PyYAML") from exc
        payload = yaml.safe_load(p.read_text(encoding="utf-8"))
        return payload or {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload or {}


def build_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Protocol-hardened evaluation runner for prunedAttentionGRU")
    parser.set_defaults(**defaults)

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices=DATASET_CHOICES, required=("dataset" not in defaults))
    parser.add_argument("--batchsize", type=int, default=int(defaults.get("batchsize", 128)))
    parser.add_argument("--learningrate", type=float, default=float(defaults.get("learningrate", 1e-3)))
    parser.add_argument("--epochs", type=int, default=int(defaults.get("epochs", 100)))
    parser.add_argument("--hidden-size", type=int, default=int(defaults.get("hidden_size", 128)))
    parser.add_argument("--attention-dim", type=int, default=int(defaults.get("attention_dim", 32)))
    parser.add_argument(
        "--model-variant",
        type=str,
        choices=list(MODEL_VARIANTS),
        default=str(defaults.get("model_variant", "baseline")),
    )
    parser.add_argument(
        "--temporal-pooling",
        type=str,
        choices=list(TEMPORAL_POOLING_CHOICES),
        default=str(defaults.get("temporal_pooling", "attn")),
    )
    parser.add_argument("--freq-feature-dim", type=int, default=int(defaults.get("freq_feature_dim", 64)))
    parser.add_argument("--fusion-hidden-dim", type=int, default=int(defaults.get("fusion_hidden_dim", 64)))
    parser.add_argument(
        "--freq-use-abs",
        type=str,
        default=str(defaults.get("freq_use_abs", "false")),
        help="Apply abs() before temporal summary; keep false for standardized feature inputs.",
    )
    parser.add_argument("--freq-eps", type=float, default=float(defaults.get("freq_eps", 1e-8)))

    parser.add_argument("--results-root", type=str, default=str(defaults.get("results_root", "results")))
    parser.add_argument(
        "--experiment-family",
        type=str,
        choices=["protocol", "paper_repro", "exploratory"],
        default=str(defaults.get("experiment_family", "protocol")),
    )
    parser.add_argument("--experiment-tag", type=str, default=str(defaults.get("experiment_tag", "")))
    parser.add_argument(
        "--experiment-group",
        type=str,
        default=str(defaults.get("experiment_group", "")),
        help="Sweep family / grouping label. Used primarily for exploratory runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(defaults.get("output_dir", "")),
        help="Legacy path override. Prefer --results-root with experiment metadata flags.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=str(defaults.get("run_name", "")),
        help="Legacy alias used as an experiment tag when present.",
    )
    parser.add_argument("--device", type=str, default=str(defaults.get("device", "auto")))
    parser.add_argument("--num-workers", type=int, default=int(defaults.get("num_workers", 0)))
    parser.add_argument("--log-every", type=int, default=int(defaults.get("log_every", 10)))

    parser.add_argument("--seed-list", type=str, default=str(defaults.get("seed_list", "42,52,62")))
    parser.add_argument("--mixup-probs", type=str, default=str(defaults.get("mixup_probs", "0.3,0.7")))
    parser.add_argument("--use-mixup", type=str, default=str(defaults.get("use_mixup", "true")))
    parser.add_argument("--augment-train", type=str, default=str(defaults.get("augment_train", "true")))
    parser.add_argument("--augment-gaussian", type=str, default=str(defaults.get("augment_gaussian", "true")))
    parser.add_argument(
        "--augment-paper-gaussian",
        type=str,
        default=str(defaults.get("augment_paper_gaussian", "false")),
    )
    parser.add_argument("--augment-shift", type=str, default=str(defaults.get("augment_shift", "true")))
    parser.add_argument("--shift-steps", type=int, default=int(defaults.get("shift_steps", 10)))
    parser.add_argument(
        "--run-planA",
        "--run-plana",
        dest="run_planA",
        type=str,
        default=str(defaults.get("run_planA", "false")),
    )
    parser.add_argument(
        "--planA-output-dir",
        "--plana-output-dir",
        dest="planA_output_dir",
        type=str,
        default=str(defaults.get("planA_output_dir", "")),
        help="Optional fixed output directory for Plan A. When empty, a timestamped directory is created.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        type=str,
        default=str(defaults.get("overwrite_output_dir", "false")),
        help="Allow Plan A to delete an existing output directory before rerunning.",
    )
    parser.add_argument(
        "--planA-include-baseline-mixup",
        "--plana-include-baseline-mixup",
        dest="planA_include_baseline_mixup",
        type=str,
        default=str(defaults.get("planA_include_baseline_mixup", "false")),
        help="Include baseline+mixup as an extra Plan A cell.",
    )

    parser.add_argument("--split-mode", type=str, choices=["predefined", "random", "group"], default=str(defaults.get("split_mode", "predefined")))
    parser.add_argument("--group-key", type=str, choices=["auto", "subject", "session", "environment"], default=str(defaults.get("group_key", "auto")))
    parser.add_argument("--train-ratio", type=float, default=float(defaults.get("train_ratio", 0.7)))
    parser.add_argument("--val-ratio", type=float, default=float(defaults.get("val_ratio", 0.1)))
    parser.add_argument("--test-ratio", type=float, default=float(defaults.get("test_ratio", 0.2)))
    parser.add_argument("--stratified-random-split", type=str, default=str(defaults.get("stratified_random_split", "true")))
    parser.add_argument("--reuse-splits", type=str, default=str(defaults.get("reuse_splits", "true")))
    parser.add_argument("--metadata-path", type=str, default=defaults.get("metadata_path", None))

    parser.add_argument("--run-robustness", type=str, default=str(defaults.get("run_robustness", "false")))
    parser.add_argument("--corruptions", type=str, default=str(defaults.get("corruptions", "gaussian_noise,frame_dropout,subcarrier_dropout,temporal_shift")))
    parser.add_argument("--severity-levels", type=str, default=str(defaults.get("severity_levels", "")))

    parser.add_argument("--hash-check-mode", type=str, choices=["quick", "none"], default=str(defaults.get("hash_check_mode", "quick")))
    parser.add_argument("--save-confusion-png", type=str, default=str(defaults.get("save_confusion_png", "true")))

    parser.add_argument("--latency-warmup", type=int, default=int(defaults.get("latency_warmup", 10)))
    parser.add_argument("--latency-iters", type=int, default=int(defaults.get("latency_iters", 30)))
    parser.add_argument(
        "--run-attention-diagnostics",
        type=str,
        default=str(defaults.get("run_attention_diagnostics", "false")),
        help="Export attention weights and run entropy/shuffle diagnostics when temporal pooling is attn.",
    )
    parser.add_argument(
        "--attention-max-export-samples",
        type=int,
        default=int(defaults.get("attention_max_export_samples", 0)),
        help="Maximum number of test samples to export attention weights for (0 means all).",
    )
    parser.add_argument(
        "--attention-shuffle-eval",
        type=str,
        default=str(defaults.get("attention_shuffle_eval", "true")),
        help="Run control evaluation by shuffling learned attention weights per sample before aggregation.",
    )
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0):
        raise ValueError("split ratios must be non-negative")
    if ratios.sum() <= 0:
        raise ValueError("split ratios sum must be > 0")
    ratios = ratios / ratios.sum()
    return float(ratios[0]), float(ratios[1]), float(ratios[2])


def apply_mixup_train_only(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    enable_mixup: bool,
    alpha: float,
    phase: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if enable_mixup and str(phase) != "train":
        raise RuntimeError("mixup may only be applied during training")
    if not enable_mixup:
        return inputs, labels

    mixed_inputs, label_a, label_b, lam = mixup(inputs, labels, alpha)
    lam_float = float(lam.item()) if hasattr(lam, "item") else float(lam)
    targets = lam_float * label_a + (1.0 - lam_float) * label_b
    return mixed_inputs, targets


def split_file_fingerprint(split_file: Path) -> str:
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    core = {
        "train": [int(v) for v in payload.get("train", [])],
        "val": [int(v) for v in payload.get("val", [])],
        "test": [int(v) for v in payload.get("test", [])],
    }
    normalized = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: Sequence[str],
    corruption: Optional[str] = None,
    severity: float = 0.0,
    seed: int = 0,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            if corruption is not None:
                batch_seed = int(seed * 1000003 + batch_idx * 7919 + int(severity * 1_000_000))
                inputs = apply_corruption(inputs, corruption=corruption, severity=severity, seed=batch_seed)

            logits = model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = criterion(logits, labels)
            total_loss += float(loss.item() * inputs.size(0))

            pred_idx = torch.argmax(logits, dim=1)
            true_idx = torch.argmax(labels, dim=1)
            y_pred.extend(pred_idx.detach().cpu().numpy().tolist())
            y_true.extend(true_idx.detach().cpu().numpy().tolist())

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    metrics = classification_metrics(y_true_np, y_pred_np, num_classes=num_classes, class_names=class_names)
    metrics["loss"] = float(total_loss / max(len(y_true), 1))
    return metrics


def _forward_with_aux(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    attention_override: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if attention_override is not None and hasattr(model, "forward_with_aux"):
        try:
            logits, aux = model.forward_with_aux(inputs, attention_override=attention_override)  # type: ignore[attr-defined]
            return logits, aux
        except TypeError:
            pass

    if hasattr(model, "forward_with_aux"):
        logits, aux = model.forward_with_aux(inputs)  # type: ignore[attr-defined]
    else:
        logits = model(inputs)
        aux = {}

    return logits, aux


def run_attention_diagnostics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: Sequence[str],
    seed: int,
    save_dir: Path,
    max_export_samples: int = 0,
    run_shuffle_eval: bool = True,
) -> Optional[Dict[str, Any]]:
    model.eval()

    collected_weights: List[np.ndarray] = []
    sample_summary_rows: List[Dict[str, Any]] = []
    total_seen = 0

    shuffle_y_true: List[int] = []
    shuffle_y_pred: List[int] = []
    shuffle_total_loss = 0.0
    shuffle_total_count = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            logits, aux = _forward_with_aux(model=model, inputs=inputs)
            attn_weights = aux.get("attention_weights")

            if attn_weights is None:
                continue

            attn_np = attn_weights.detach().cpu().numpy().astype(np.float64)
            batch_size, time_steps = attn_np.shape

            if max_export_samples <= 0 or total_seen < max_export_samples:
                remaining = attn_np.shape[0]
                if max_export_samples > 0:
                    remaining = min(remaining, max_export_samples - total_seen)
                if remaining > 0:
                    export_slice = attn_np[:remaining]
                    collected_weights.append(export_slice)
                    top_idx = np.argmax(export_slice, axis=1)
                    top_val = np.max(export_slice, axis=1)
                    for row_i in range(export_slice.shape[0]):
                        sample_summary_rows.append(
                            {
                                "sample_index": int(total_seen + row_i),
                                "batch_index": int(batch_idx),
                                "top_timestep": int(top_idx[row_i]),
                                "top_weight": float(top_val[row_i]),
                            }
                        )
                    total_seen += int(remaining)

            if run_shuffle_eval:
                random_scores = torch.rand_like(attn_weights)
                permutation = torch.argsort(random_scores, dim=1)
                shuffled_weights = torch.gather(attn_weights, dim=1, index=permutation)

                shuffled_logits, _ = _forward_with_aux(
                    model=model,
                    inputs=inputs,
                    attention_override=shuffled_weights,
                )
                shuffle_loss = criterion(shuffled_logits, labels)
                shuffle_total_loss += float(shuffle_loss.item() * inputs.size(0))
                shuffle_total_count += int(inputs.size(0))

                pred_idx = torch.argmax(shuffled_logits, dim=1)
                true_idx = torch.argmax(labels, dim=1)
                shuffle_y_pred.extend(pred_idx.detach().cpu().numpy().tolist())
                shuffle_y_true.extend(true_idx.detach().cpu().numpy().tolist())

    if not collected_weights:
        return None

    all_weights = np.concatenate(collected_weights, axis=0)
    np.save(save_dir / "attention_weights.npy", all_weights)
    write_csv(
        save_dir / "attention_weight_sample_summary.csv",
        sample_summary_rows,
        fieldnames=["sample_index", "batch_index", "top_timestep", "top_weight"],
    )

    eps = 1e-12
    clipped = np.clip(all_weights, eps, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    time_steps = int(all_weights.shape[1])
    uniform_entropy = float(np.log(max(time_steps, 1)))
    normalized_entropy = entropy / max(uniform_entropy, eps)

    mean_profile = all_weights.mean(axis=0)
    std_across_samples = all_weights.std(axis=0, ddof=0)
    l1_to_mean = np.abs(all_weights - mean_profile[None, :]).mean(axis=1)
    top_idx_all = np.argmax(all_weights, axis=1)
    top_hist = np.bincount(top_idx_all, minlength=time_steps)

    diag: Dict[str, Any] = {
        "seed": int(seed),
        "num_samples": int(all_weights.shape[0]),
        "time_steps": time_steps,
        "uniform_entropy_nats": uniform_entropy,
        "entropy_mean_nats": float(entropy.mean()),
        "entropy_std_nats": float(entropy.std(ddof=0)),
        "normalized_entropy_mean": float(normalized_entropy.mean()),
        "normalized_entropy_std": float(normalized_entropy.std(ddof=0)),
        "mean_sample_l1_to_global_profile": float(l1_to_mean.mean()),
        "std_sample_l1_to_global_profile": float(l1_to_mean.std(ddof=0)),
        "mean_timestep_std_across_samples": float(std_across_samples.mean()),
        "max_timestep_std_across_samples": float(std_across_samples.max()),
        "top_timestep_unique_count": int(np.count_nonzero(top_hist)),
        "top_timestep_unique_ratio": float(np.count_nonzero(top_hist) / max(time_steps, 1)),
        "top_timestep_histogram": top_hist.astype(int).tolist(),
        "mean_attention_profile": mean_profile.astype(float).tolist(),
    }

    if run_shuffle_eval and shuffle_total_count > 0:
        y_true_np = np.asarray(shuffle_y_true, dtype=np.int64)
        y_pred_np = np.asarray(shuffle_y_pred, dtype=np.int64)
        shuffle_metrics = classification_metrics(
            y_true_np,
            y_pred_np,
            num_classes=num_classes,
            class_names=class_names,
        )
        shuffle_metrics["loss"] = float(shuffle_total_loss / max(shuffle_total_count, 1))
        diag["shuffle_control"] = {
            "accuracy": float(shuffle_metrics["accuracy"]),
            "macro_f1": float(shuffle_metrics["macro_f1"]),
            "loss": float(shuffle_metrics["loss"]),
            "num_samples": int(shuffle_metrics["num_samples"]),
        }

    write_json(save_dir / "attention_diagnostics.json", diag)
    return diag


def collect_model_summary(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    split_name: str,
    checkpoint_tag: str = "best",
) -> Dict[str, Any]:
    summary = dict(model_metadata(model))
    gate_values: List[torch.Tensor] = []
    gate_dim: Optional[int] = None

    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.float().to(device)

            if hasattr(model, "forward_with_aux"):
                _, aux = model.forward_with_aux(inputs)  # type: ignore[attr-defined]
                freq_signature = aux.get("freq_signature")
                if freq_signature is not None:
                    summary["raw_frequency_signature_dim"] = int(freq_signature.shape[-1])

                gate = aux.get("gate")
                if gate is not None:
                    gate_dim = int(gate.shape[-1])
                    gate_values.append(gate.detach().cpu().reshape(-1))
            else:
                _ = model(inputs)

    freq_dim = summary.get("raw_frequency_signature_dim", None)
    if freq_dim is not None:
        summary["raw_frequency_signature_dim"] = int(freq_dim)

    if gate_values:
        gate_cat = torch.cat(gate_values, dim=0)
        summary["gate_statistics"] = {
            "split": str(split_name),
            "checkpoint": str(checkpoint_tag),
            "aggregation": "all_samples_x_gate_dims_pooled",
            "gate_dim": gate_dim,
            "num_values": int(gate_cat.numel()),
            "mean": float(gate_cat.mean().item()),
            "std": float(gate_cat.std(unbiased=False).item()),
            "min": float(gate_cat.min().item()),
            "max": float(gate_cat.max().item()),
        }

    return summary


def train_one_seed(
    seed: int,
    args: Dict[str, Any],
    bundle,
    run_dir: Path,
    results_root: Path,
    splits_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    seed_dir = ensure_dir(run_dir / "seeds" / f"seed_{seed}")
    figures_dir = ensure_dir(seed_dir / "figures")

    split_file = splits_dir / f"split_seed{seed}.json"
    split, split_meta = generate_or_load_split(
        bundle=bundle,
        split_file=split_file,
        split_mode=args["split_mode"],
        group_key=args["group_key"],
        seed=seed,
        train_ratio=args["train_ratio"],
        val_ratio=args["val_ratio"],
        test_ratio=args["test_ratio"],
        stratified_random=args["stratified_random_split"],
        reuse_existing=args["reuse_splits"],
    )

    leakage = leakage_checks(bundle, split=split, hash_mode=args["hash_check_mode"])
    if split.warnings:
        print(f"[seed={seed}] split warnings: {split.warnings}")
    if leakage["warnings"]:
        print(f"[seed={seed}] leakage warnings: {leakage['warnings']}")

    train_loader, val_loader, test_loader = build_dataloaders(
        bundle=bundle,
        split=split,
        train_batch_size=args["batchsize"],
        eval_batch_size=args["batchsize"],
        num_workers=args["num_workers"],
        seed=seed,
        augment_train=args["augment_train"],
        augment_gaussian=args["augment_gaussian"],
        augment_paper_gaussian=args["augment_paper_gaussian"],
        augment_shift=args["augment_shift"],
        shift_steps=args["shift_steps"],
    )

    if len(train_loader.dataset) == 0 or len(test_loader.dataset) == 0:
        raise RuntimeError("empty train/test split")
    if len(val_loader.dataset) == 0:
        raise RuntimeError("validation split is empty; refusing to use test split for checkpoint selection")

    eval_loader = val_loader
    eval_split_name = "val"

    model = build_model(
        model_variant=args["model_variant"],
        input_dim=bundle.input_size,
        hidden_dim=args["hidden_size"],
        attention_dim=args["attention_dim"],
        output_dim=bundle.num_classes,
        freq_feature_dim=args["freq_feature_dim"],
        fusion_hidden_dim=args["fusion_hidden_dim"],
        freq_use_abs=args["freq_use_abs"],
        freq_eps=args["freq_eps"],
        temporal_pooling=args["temporal_pooling"],
    ).to(device)
    criterion = SoftTargetCrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learningrate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args["epochs"])

    mixup_probs = args["mixup_probs"]
    if len(mixup_probs) != 2:
        raise ValueError("mixup_probs must contain 2 values")
    prob_sum = float(mixup_probs[0] + mixup_probs[1])
    p_mixup = float(mixup_probs[0] / max(prob_sum, 1e-12)) if args["use_mixup"] else 0.0

    history_rows: List[Dict[str, Any]] = []
    best_state = None
    best_eval_macro_f1 = -1.0
    best_eval_accuracy = 0.0

    train_start = time.perf_counter()
    for epoch in range(args["epochs"]):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            use_mixup_batch = bool(args["use_mixup"]) and (np.random.rand() < p_mixup)
            inputs, targets = apply_mixup_train_only(
                inputs,
                labels,
                enable_mixup=use_mixup_batch,
                alpha=1.0,
                phase="train",
            )

            logits = model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            pred_idx = torch.argmax(logits, dim=1)
            true_idx = torch.argmax(targets, dim=1)
            running_correct += int((pred_idx == true_idx).sum().item())
            running_total += int(inputs.size(0))
            running_loss += float(loss.item() * inputs.size(0))

        scheduler.step()

        train_loss = running_loss / max(running_total, 1)
        train_acc = float(running_correct / max(running_total, 1))

        eval_metrics = evaluate(
            model=model,
            loader=eval_loader,
            criterion=criterion,
            device=device,
            num_classes=bundle.num_classes,
            class_names=bundle.class_names,
            seed=seed,
        )

        history_rows.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "eval_loss": float(eval_metrics["loss"]),
                "eval_accuracy": float(eval_metrics["accuracy"]),
                "eval_macro_f1": float(eval_metrics["macro_f1"]),
            }
        )

        if float(eval_metrics["macro_f1"]) > best_eval_macro_f1:
            best_eval_macro_f1 = float(eval_metrics["macro_f1"])
            best_eval_accuracy = float(eval_metrics["accuracy"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if args["log_every"] > 0 and (
            (epoch + 1) % args["log_every"] == 0 or epoch == 0 or (epoch + 1) == args["epochs"]
        ):
            print(
                f"Epoch {epoch + 1}/{args['epochs']} "
                f"train_acc={train_acc:.4f} train_loss={train_loss:.4f} "
                f"{eval_split_name}_acc={eval_metrics['accuracy']:.4f} {eval_split_name}_macro_f1={eval_metrics['macro_f1']:.4f}"
            )

    train_seconds = time.perf_counter() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_start = time.perf_counter()
    clean_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=bundle.num_classes,
        class_names=bundle.class_names,
        seed=seed,
    )
    eval_seconds = time.perf_counter() - eval_start
    checkpoint_path = seed_dir / "checkpoint_best.pt"
    torch.save(
        {
            "model_variant": args["model_variant"],
            "temporal_pooling": args["temporal_pooling"],
            "state_dict": model.state_dict(),
            "input_size": int(bundle.input_size),
            "num_classes": int(bundle.num_classes),
            "hidden_size": int(args["hidden_size"]),
            "attention_dim": int(args["attention_dim"]),
            "freq_feature_dim": int(args["freq_feature_dim"]),
            "fusion_hidden_dim": int(args["fusion_hidden_dim"]),
        },
        checkpoint_path,
    )

    model_summary = collect_model_summary(
        model=model,
        loader=test_loader,
        device=device,
        split_name="test_clean",
        checkpoint_tag="best",
    )

    attention_diag: Optional[Dict[str, Any]] = None
    if args["run_attention_diagnostics"] and str(args["temporal_pooling"]).strip().lower() == "attn":
        attention_diag = run_attention_diagnostics(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=bundle.num_classes,
            class_names=bundle.class_names,
            seed=seed,
            save_dir=seed_dir,
            max_export_samples=int(args["attention_max_export_samples"]),
            run_shuffle_eval=bool(args["attention_shuffle_eval"]),
        )

    params_info = count_parameters(model)
    sample_batch = next(iter(test_loader))[0]
    latency = measure_inference_latency(
        model=model,
        sample_batch=sample_batch,
        device=device,
        warmup=args["latency_warmup"],
        iters=args["latency_iters"],
    )

    confusion = clean_metrics["confusion_matrix"]
    np.save(seed_dir / "confusion_matrix.npy", confusion)
    np.save(seed_dir / "confusion_matrix_row_normalized.npy", normalize_confusion_rows(confusion))

    if args["save_confusion_png"]:
        maybe_save_confusion_png(
            confusion=confusion,
            class_names=bundle.class_names,
            path=figures_dir / "confusion_matrix.png",
            normalize=False,
            title=f"Confusion (seed={seed})",
        )
        maybe_save_confusion_png(
            confusion=confusion,
            class_names=bundle.class_names,
            path=figures_dir / "confusion_matrix_row_normalized.png",
            normalize=True,
            title=f"Confusion Row-Normalized (seed={seed})",
        )

    metrics_row = {
        "seed": int(seed),
        "model_variant": str(model_summary.get("model_variant", args["model_variant"])),
        "fusion_type": str(model_summary.get("fusion_type", "none")),
        "temporal_pooling": str(model_summary.get("temporal_pooling", args["temporal_pooling"])),
        "frequency_use_abs": model_summary.get("frequency_use_abs"),
        "raw_frequency_signature_dim": model_summary.get("raw_frequency_signature_dim"),
        "projected_frequency_feature_dim": model_summary.get("projected_frequency_feature_dim"),
        "split_mode_requested": split.mode_requested,
        "split_mode_used": split.mode_used,
        "group_key_requested": split.group_key_requested,
        "group_key_used": split.group_key_used,
        "eval_split_used_for_checkpoint": eval_split_name,
        "val_accuracy_best": float(best_eval_accuracy),
        "val_macro_f1_best": float(best_eval_macro_f1),
        "accuracy": float(clean_metrics["accuracy"]),
        "macro_f1": float(clean_metrics["macro_f1"]),
        "loss": float(clean_metrics["loss"]),
        "num_samples": int(clean_metrics["num_samples"]),
        "augment_train": bool(args["augment_train"]),
        "augment_gaussian": bool(args["augment_gaussian"]),
        "augment_paper_gaussian": bool(args["augment_paper_gaussian"]),
        "augment_shift": bool(args["augment_shift"]),
        "shift_steps": int(args["shift_steps"]),
        "use_mixup": bool(args["use_mixup"]),
        "train_time_sec": float(train_seconds),
        "eval_time_sec": float(eval_seconds),
        "checkpoint_path": relative_to_results_root(checkpoint_path, results_root),
        **params_info,
        **latency,
    }

    if "gate_statistics" in model_summary:
        metrics_row["gate_mean"] = float(model_summary["gate_statistics"]["mean"])
        metrics_row["gate_std"] = float(model_summary["gate_statistics"]["std"])
        metrics_row["gate_num_values"] = int(model_summary["gate_statistics"]["num_values"])

    if attention_diag is not None:
        metrics_row["attn_entropy_norm_mean"] = float(attention_diag["normalized_entropy_mean"])
        metrics_row["attn_entropy_norm_std"] = float(attention_diag["normalized_entropy_std"])
        metrics_row["attn_top_timestep_unique_ratio"] = float(attention_diag["top_timestep_unique_ratio"])
        metrics_row["attn_l1_to_mean_profile"] = float(attention_diag["mean_sample_l1_to_global_profile"])
        shuffle_control = attention_diag.get("shuffle_control")
        if isinstance(shuffle_control, dict):
            metrics_row["shuffle_accuracy"] = float(shuffle_control["accuracy"])
            metrics_row["shuffle_macro_f1"] = float(shuffle_control["macro_f1"])
            metrics_row["shuffle_loss"] = float(shuffle_control["loss"])
            metrics_row["delta_macro_f1_shuffle"] = float(metrics_row["macro_f1"] - metrics_row["shuffle_macro_f1"])
            metrics_row["delta_accuracy_shuffle"] = float(metrics_row["accuracy"] - metrics_row["shuffle_accuracy"])

    write_json(seed_dir / "metrics.json", metrics_row)
    write_json(seed_dir / "model_summary.json", model_summary)
    write_csv(seed_dir / "metrics.csv", [metrics_row], fieldnames=list(metrics_row.keys()))
    write_csv(
        seed_dir / "per_class_f1.csv",
        clean_metrics["per_class_rows"],
        fieldnames=["class_index", "class_name", "f1", "support"],
    )
    write_csv(
        seed_dir / "training_history.csv",
        history_rows,
        fieldnames=["epoch", "train_loss", "train_accuracy", "eval_loss", "eval_accuracy", "eval_macro_f1"],
    )
    write_json(seed_dir / "split_summary.json", split_meta)
    write_json(seed_dir / "leakage_report.json", leakage)

    robustness_rows: List[Dict[str, Any]] = []
    if args["run_robustness"]:
        for corruption in args["corruptions"]:
            levels = resolve_severity_levels(corruption, args["severity_levels"])
            for severity in levels:
                corr_metrics = evaluate(
                    model=model,
                    loader=test_loader,
                    criterion=criterion,
                    device=device,
                    num_classes=bundle.num_classes,
                    class_names=bundle.class_names,
                    corruption=corruption,
                    severity=float(severity),
                    seed=seed,
                )
                robustness_rows.append(
                    {
                        "seed": int(seed),
                        "corruption": str(corruption),
                        "severity": float(severity),
                        "accuracy": float(corr_metrics["accuracy"]),
                        "macro_f1": float(corr_metrics["macro_f1"]),
                        "loss": float(corr_metrics["loss"]),
                        "num_samples": int(corr_metrics["num_samples"]),
                    }
                )

        if robustness_rows:
            write_csv(
                seed_dir / "robustness_results.csv",
                robustness_rows,
                fieldnames=["seed", "corruption", "severity", "accuracy", "macro_f1", "loss", "num_samples"],
            )

    return {
        "seed_metrics": metrics_row,
        "model_summary": model_summary,
        "clean_per_class_rows": clean_metrics["per_class_rows"],
        "clean_confusion": confusion,
        "attention_diag": attention_diag,
        "robustness_rows": robustness_rows,
        "split_file": str(split_meta.get("split_file", split_file)),
        "split_counts": {
            "train": int(len(split.train)),
            "val": int(len(split.val)),
            "test": int(len(split.test)),
        },
        "eval_split_name": eval_split_name,
    }


def aggregate_outputs(run_dir: Path, bundle, seed_outputs: Sequence[Dict[str, Any]], save_confusion_png: bool) -> None:
    aggregate_dir = ensure_dir(run_dir / "aggregate")
    robustness_dir = ensure_dir(run_dir / "robustness")
    seed_rows = [entry["seed_metrics"] for entry in seed_outputs]
    write_csv(aggregate_dir / "seed_metrics.csv", seed_rows, fieldnames=list(seed_rows[0].keys()))

    summary = aggregate_mean_std(
        seed_rows,
        keys=["accuracy", "macro_f1", "loss", "train_time_sec", "latency_ms_per_batch", "throughput_samples_per_sec"],
    )
    summary_rows = []
    for metric, stats in summary.items():
        summary_rows.append({"metric": metric, "mean": stats["mean"], "std": stats["std"]})
    write_csv(aggregate_dir / "summary_metrics.csv", summary_rows, fieldnames=["metric", "mean", "std"])

    per_class_values = []
    for entry in seed_outputs:
        ordered = sorted(entry["clean_per_class_rows"], key=lambda row: int(row["class_index"]))
        per_class_values.append([float(row["f1"]) for row in ordered])
    per_class_values_np = np.asarray(per_class_values, dtype=np.float64)

    per_class_rows = []
    for idx, name in enumerate(bundle.class_names):
        values = per_class_values_np[:, idx]
        per_class_rows.append(
            {
                "class_index": int(idx),
                "class_name": name,
                "f1_mean": float(values.mean()),
                "f1_std": float(values.std(ddof=0)),
            }
        )
    write_csv(
        aggregate_dir / "per_class_f1_summary.csv",
        per_class_rows,
        fieldnames=["class_index", "class_name", "f1_mean", "f1_std"],
    )

    confusion_stack = np.stack([entry["clean_confusion"] for entry in seed_outputs], axis=0)
    confusion_mean = confusion_stack.mean(axis=0)
    confusion_mean_norm = normalize_confusion_rows(confusion_mean)
    np.save(aggregate_dir / "confusion_matrix_mean.npy", confusion_mean)
    np.save(aggregate_dir / "confusion_matrix_mean_row_normalized.npy", confusion_mean_norm)

    if save_confusion_png:
        maybe_save_confusion_png(
            confusion=confusion_mean,
            class_names=bundle.class_names,
            path=aggregate_dir / "confusion_matrix_mean.png",
            normalize=False,
            title="Mean Confusion Across Seeds",
        )
        maybe_save_confusion_png(
            confusion=confusion_mean,
            class_names=bundle.class_names,
            path=aggregate_dir / "confusion_matrix_mean_row_normalized.png",
            normalize=True,
            title="Mean Row-Normalized Confusion Across Seeds",
        )

    attention_rows = []
    for entry in seed_outputs:
        diag = entry.get("attention_diag")
        if not diag:
            continue
        row = {
            "seed": int(diag["seed"]),
            "num_samples": int(diag["num_samples"]),
            "time_steps": int(diag["time_steps"]),
            "normalized_entropy_mean": float(diag["normalized_entropy_mean"]),
            "normalized_entropy_std": float(diag["normalized_entropy_std"]),
            "mean_sample_l1_to_global_profile": float(diag["mean_sample_l1_to_global_profile"]),
            "top_timestep_unique_ratio": float(diag["top_timestep_unique_ratio"]),
        }
        shuffle = diag.get("shuffle_control")
        if isinstance(shuffle, dict):
            row["shuffle_accuracy"] = float(shuffle["accuracy"])
            row["shuffle_macro_f1"] = float(shuffle["macro_f1"])
            row["shuffle_loss"] = float(shuffle["loss"])
        attention_rows.append(row)

    if attention_rows:
        write_csv(
            aggregate_dir / "attention_diagnostics_summary.csv",
            attention_rows,
            fieldnames=list(attention_rows[0].keys()),
        )

    all_robustness_rows: List[Dict[str, Any]] = []
    for entry in seed_outputs:
        all_robustness_rows.extend(entry["robustness_rows"])

    if all_robustness_rows:
        write_csv(
            robustness_dir / "robustness_results.csv",
            all_robustness_rows,
            fieldnames=["seed", "corruption", "severity", "accuracy", "macro_f1", "loss", "num_samples"],
        )

        grouped: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
        for row in all_robustness_rows:
            key = (str(row["corruption"]), float(row["severity"]))
            grouped.setdefault(key, []).append(row)

        summary_rows = []
        for (corruption, severity), rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            acc = np.asarray([float(r["accuracy"]) for r in rows], dtype=np.float64)
            f1 = np.asarray([float(r["macro_f1"]) for r in rows], dtype=np.float64)
            summary_rows.append(
                {
                    "corruption": corruption,
                    "severity": float(severity),
                    "accuracy_mean": float(acc.mean()),
                    "accuracy_std": float(acc.std(ddof=0)),
                    "macro_f1_mean": float(f1.mean()),
                    "macro_f1_std": float(f1.std(ddof=0)),
                }
            )
        write_csv(
            robustness_dir / "robustness_summary.csv",
            summary_rows,
            fieldnames=["corruption", "severity", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"],
        )


def normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(raw)

    cfg["run_planA"] = str2bool(cfg["run_planA"])
    cfg["overwrite_output_dir"] = str2bool(cfg["overwrite_output_dir"])
    cfg["planA_include_baseline_mixup"] = str2bool(cfg["planA_include_baseline_mixup"])
    cfg["use_mixup"] = str2bool(cfg["use_mixup"])
    cfg["augment_train"] = str2bool(cfg["augment_train"])
    cfg["augment_gaussian"] = str2bool(cfg["augment_gaussian"])
    cfg["augment_paper_gaussian"] = str2bool(cfg["augment_paper_gaussian"])
    cfg["augment_shift"] = str2bool(cfg["augment_shift"])
    cfg["freq_use_abs"] = str2bool(cfg["freq_use_abs"])
    cfg["stratified_random_split"] = str2bool(cfg["stratified_random_split"])
    cfg["reuse_splits"] = str2bool(cfg["reuse_splits"])
    cfg["run_robustness"] = str2bool(cfg["run_robustness"])
    cfg["save_confusion_png"] = str2bool(cfg["save_confusion_png"])
    cfg["run_attention_diagnostics"] = str2bool(cfg["run_attention_diagnostics"])
    cfg["attention_shuffle_eval"] = str2bool(cfg["attention_shuffle_eval"])

    cfg["model_variant"] = str(cfg["model_variant"]).strip().lower()
    if cfg["model_variant"] not in MODEL_VARIANTS:
        raise ValueError(f"unsupported model_variant: {cfg['model_variant']}")

    cfg["temporal_pooling"] = str(cfg["temporal_pooling"]).strip().lower()
    if cfg["temporal_pooling"] not in TEMPORAL_POOLING_CHOICES:
        raise ValueError(f"unsupported temporal_pooling: {cfg['temporal_pooling']}")

    cfg["train_ratio"], cfg["val_ratio"], cfg["test_ratio"] = resolve_ratios(
        float(cfg["train_ratio"]), float(cfg["val_ratio"]), float(cfg["test_ratio"])
    )

    seed_raw = cfg["seed_list"]
    if isinstance(seed_raw, str):
        cfg["seed_list"] = parse_int_list(seed_raw)
    elif isinstance(seed_raw, (list, tuple)):
        cfg["seed_list"] = [int(v) for v in seed_raw]
    else:
        raise ValueError("seed_list must be string or list")

    mixup_raw = cfg["mixup_probs"]
    if isinstance(mixup_raw, str):
        cfg["mixup_probs"] = parse_float_list(mixup_raw)
    elif isinstance(mixup_raw, (list, tuple)):
        cfg["mixup_probs"] = [float(v) for v in mixup_raw]
    else:
        raise ValueError("mixup_probs must be string or list")

    corruption_raw = cfg["corruptions"]
    if isinstance(corruption_raw, str):
        cfg["corruptions"] = [name.strip() for name in corruption_raw.split(",") if name.strip()]
    elif isinstance(corruption_raw, (list, tuple)):
        cfg["corruptions"] = [str(name).strip() for name in corruption_raw if str(name).strip()]
    else:
        raise ValueError("corruptions must be string or list")

    for name in cfg["corruptions"]:
        if name not in SUPPORTED_CORRUPTIONS:
            raise ValueError(f"unsupported corruption: {name}")

    severity_raw = cfg["severity_levels"]
    if isinstance(severity_raw, str):
        cfg["severity_levels"] = parse_float_list(severity_raw) if severity_raw.strip() else []
    elif isinstance(severity_raw, (list, tuple)):
        cfg["severity_levels"] = [float(v) for v in severity_raw]
    else:
        raise ValueError("severity_levels must be string or list")

    cfg["attention_max_export_samples"] = int(cfg["attention_max_export_samples"])
    if cfg["attention_max_export_samples"] < 0:
        raise ValueError("attention_max_export_samples must be >= 0")

    cfg["results_root"] = str(cfg["results_root"]).strip() or "results"
    cfg["experiment_family"] = str(cfg["experiment_family"]).strip().lower() or "protocol"
    cfg["experiment_tag"] = str(cfg["experiment_tag"]).strip()
    cfg["experiment_group"] = str(cfg["experiment_group"]).strip()
    cfg["output_dir"] = str(cfg["output_dir"]).strip()
    cfg["run_name"] = str(cfg["run_name"]).strip()
    cfg["planA_output_dir"] = str(cfg["planA_output_dir"]).strip()
    cfg["shift_steps"] = int(cfg["shift_steps"])

    if cfg["shift_steps"] < 0:
        raise ValueError("shift_steps must be >= 0")
    if cfg["augment_paper_gaussian"] and not cfg["augment_gaussian"]:
        raise ValueError("augment_paper_gaussian=true requires augment_gaussian=true")
    if cfg["augment_shift"] and not cfg["augment_train"]:
        raise ValueError("shift augmentation requested while augment_train=false")
    if not cfg["augment_train"]:
        cfg["augment_gaussian"] = False
        cfg["augment_paper_gaussian"] = False
        cfg["augment_shift"] = False

    return cfg


def run_planA_ablation(
    args: Dict[str, Any],
    bundle,
    device: torch.device,
    repo_root: Path,
    results_root: Path,
) -> Path:
    if str(args["dataset"]).strip().lower() != "aril":
        raise ValueError("run_planA currently supports dataset=aril only")
    if str(args["split_mode"]).strip().lower() != "predefined":
        raise ValueError("run_planA requires split_mode=predefined")

    required_seeds = [42, 52, 62]
    if [int(v) for v in args["seed_list"]] != required_seeds:
        raise ValueError("run_planA requires --seed-list 42,52,62")

    requested_output_dir = str(args.get("planA_output_dir", "")).strip()
    if requested_output_dir:
        ablation_root = Path(requested_output_dir).expanduser()
        if not ablation_root.is_absolute():
            ablation_root = (repo_root / ablation_root).resolve()
        else:
            ablation_root = ablation_root.resolve()
        if ablation_root.exists():
            if not bool(args.get("overwrite_output_dir", False)):
                raise FileExistsError(
                    f"Plan A output directory already exists: {ablation_root}. "
                    "Pass --overwrite-output-dir true to replace it."
                )
            shutil.rmtree(ablation_root)
        ablation_root = ensure_dir(ablation_root)
    else:
        ablation_root = ensure_dir(results_root / "ablations" / f"aril_planA_{timestamp_now()}")
    cells_root = ensure_dir(ablation_root / "cells")
    shared_splits_dir = ensure_dir(ablation_root / "shared_splits")

    dataset_summary = {
        "dataset": bundle.dataset_name,
        "num_samples": int(bundle.x_all.shape[0]),
        "num_classes": int(bundle.num_classes),
        "input_size": int(bundle.input_size),
        "class_names": bundle.class_names,
    }
    write_json(
        ablation_root / "ablation_metadata.json",
        {
            "ablation_name": "planA",
            "dataset": "aril",
            "timestamp": timestamp_now(),
            "base_args": to_serializable(args),
            "dataset_summary": dataset_summary,
            "environment": collect_environment_info(repo_root),
        },
    )

    cells = [
        {
            "cell_name": "aril_planA_baseline",
            "overrides": {
                "augment_train": False,
                "augment_gaussian": False,
                "augment_paper_gaussian": False,
                "augment_shift": False,
                "use_mixup": False,
            },
        },
        {
            "cell_name": "aril_planA_gaussian",
            "overrides": {
                "augment_train": True,
                "augment_gaussian": True,
                "augment_paper_gaussian": True,
                "augment_shift": False,
                "use_mixup": False,
            },
        },
        {
            "cell_name": "aril_planA_gaussian_shift",
            "overrides": {
                "augment_train": True,
                "augment_gaussian": True,
                "augment_paper_gaussian": True,
                "augment_shift": True,
                "use_mixup": False,
            },
        },
        {
            "cell_name": "aril_planA_gaussian_shift_mixup",
            "overrides": {
                "augment_train": True,
                "augment_gaussian": True,
                "augment_paper_gaussian": True,
                "augment_shift": True,
                "use_mixup": True,
            },
        },
    ]

    if bool(args.get("planA_include_baseline_mixup", False)):
        cells.append(
            {
                "cell_name": "aril_planA_baseline_mixup_cuda",
                "overrides": {
                    "augment_train": False,
                    "augment_gaussian": False,
                    "augment_paper_gaussian": False,
                    "augment_shift": False,
                    "use_mixup": True,
                },
            }
        )

    mixup_cells = [cell["cell_name"] for cell in cells if bool(cell["overrides"]["use_mixup"])]
    expected_mixup_cells = ["aril_planA_gaussian_shift_mixup"]
    if bool(args.get("planA_include_baseline_mixup", False)):
        expected_mixup_cells.append("aril_planA_baseline_mixup_cuda")
    if sorted(mixup_cells) != sorted(expected_mixup_cells):
        raise RuntimeError("Plan A guardrail violated: unexpected mixup cell configuration")

    summary_rows: List[Dict[str, Any]] = []
    reference_split_fingerprints: Optional[Dict[int, str]] = None

    for cell in cells:
        cell_name = str(cell["cell_name"])
        cell_dir = ensure_dir(cells_root / cell_name)
        cell_config_dir = ensure_dir(cell_dir / "config")

        cell_args = dict(args)
        cell_args.update(cell["overrides"])
        cell_args["split_mode"] = "predefined"
        cell_args["reuse_splits"] = True
        cell_args["run_robustness"] = False
        cell_args["shift_steps"] = int(args["shift_steps"])

        if cell_args["augment_paper_gaussian"] and not cell_args["augment_gaussian"]:
            raise RuntimeError(f"[{cell_name}] paper_gaussian=true but augment_gaussian=false")
        if cell_args["augment_shift"] and not cell_args["augment_train"]:
            raise RuntimeError(f"[{cell_name}] shift requested while augment_train=false")
        if not cell_args["augment_train"]:
            cell_args["augment_gaussian"] = False
            cell_args["augment_paper_gaussian"] = False
            cell_args["augment_shift"] = False

        write_json(cell_config_dir / "resolved_config.json", to_serializable(cell_args))

        seed_outputs = []
        for seed in cell_args["seed_list"]:
            print(f"[PlanA] {cell_name} seed={seed}")
            set_global_seed(int(seed), deterministic=True)
            out = train_one_seed(
                seed=int(seed),
                args=cell_args,
                bundle=bundle,
                run_dir=cell_dir,
                results_root=results_root,
                splits_dir=shared_splits_dir,
                device=device,
            )
            if str(out.get("eval_split_name", "")) != "val":
                raise RuntimeError("test set used for checkpoint selection; val split is required")
            seed_outputs.append(out)

        aggregate_outputs(
            run_dir=cell_dir,
            bundle=bundle,
            seed_outputs=seed_outputs,
            save_confusion_png=bool(cell_args["save_confusion_png"]),
        )

        split_paths = []
        split_fingerprints: Dict[int, str] = {}
        for seed in cell_args["seed_list"]:
            split_file = shared_splits_dir / f"split_seed{int(seed)}.json"
            if not split_file.exists():
                raise FileNotFoundError(f"missing split artifact: {split_file}")
            split_fingerprints[int(seed)] = split_file_fingerprint(split_file)
            split_paths.append(relative_to_results_root(split_file, results_root))

        if reference_split_fingerprints is None:
            reference_split_fingerprints = dict(split_fingerprints)
        elif split_fingerprints != reference_split_fingerprints:
            raise RuntimeError("different cells used different split artifacts")

        split_counts = [entry["split_counts"] for entry in seed_outputs]
        first_counts = split_counts[0]
        for counts in split_counts[1:]:
            if counts != first_counts:
                raise RuntimeError(f"inconsistent split sizes across seeds in cell {cell_name}")

        seed_rows = [entry["seed_metrics"] for entry in seed_outputs]
        test_accuracy_mean, test_accuracy_std = _mean_std([row["accuracy"] for row in seed_rows])
        test_macro_f1_mean, test_macro_f1_std = _mean_std([row["macro_f1"] for row in seed_rows])
        val_accuracy_mean, _ = _mean_std([row["val_accuracy_best"] for row in seed_rows])
        val_macro_f1_mean, _ = _mean_std([row["val_macro_f1_best"] for row in seed_rows])

        mixup_config = {
            "enabled": bool(cell_args["use_mixup"]),
            "mixup_probs": [float(v) for v in cell_args["mixup_probs"]],
            "alpha": 1.0,
        }

        summary_rows.append(
            {
                "cell_name": cell_name,
                "dataset": str(cell_args["dataset"]),
                "seed_list": [int(v) for v in cell_args["seed_list"]],
                "split_mode": str(cell_args["split_mode"]),
                "split_file": ";".join(split_paths),
                "augment_train": bool(cell_args["augment_train"]),
                "augment_gaussian": bool(cell_args["augment_gaussian"]),
                "augment_paper_gaussian": bool(cell_args["augment_paper_gaussian"]),
                "augment_shift": bool(cell_args["augment_shift"]),
                "shift_steps": int(cell_args["shift_steps"]),
                "use_mixup": bool(cell_args["use_mixup"]),
                "mixup_configuration": mixup_config,
                "num_train": int(first_counts["train"]),
                "num_val": int(first_counts["val"]),
                "num_test": int(first_counts["test"]),
                "test_accuracy_mean": test_accuracy_mean,
                "test_accuracy_std": test_accuracy_std,
                "test_macro_f1_mean": test_macro_f1_mean,
                "test_macro_f1_std": test_macro_f1_std,
                "val_accuracy_mean": val_accuracy_mean,
                "val_macro_f1_mean": val_macro_f1_mean,
            }
        )

    fieldnames = [
        "cell_name",
        "dataset",
        "seed_list",
        "split_mode",
        "split_file",
        "augment_train",
        "augment_gaussian",
        "augment_paper_gaussian",
        "augment_shift",
        "shift_steps",
        "use_mixup",
        "mixup_configuration",
        "num_train",
        "num_val",
        "num_test",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_macro_f1_mean",
        "test_macro_f1_std",
        "val_accuracy_mean",
        "val_macro_f1_mean",
    ]
    write_csv(ablation_root / "planA_ablation_summary.csv", summary_rows, fieldnames=fieldnames)
    write_json(
        ablation_root / "planA_ablation_summary.json",
        {
            "ablation_name": "planA",
            "dataset": "aril",
            "cells": summary_rows,
            "shared_split_dir": relative_to_results_root(shared_splits_dir, results_root),
            "split_fingerprints_by_seed": reference_split_fingerprints or {},
        },
    )

    return ablation_root


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    boot_args, _ = bootstrap.parse_known_args()

    defaults = load_config(boot_args.config)
    parser = build_parser(defaults)
    args_ns = parser.parse_args()

    args = normalize_config(vars(args_ns))
    device = resolve_device(args["device"])

    repo_root = Path(__file__).resolve().parent
    results_root = Path(args["results_root"]).resolve()
    write_results_readme(results_root)

    if args["run_planA"]:
        print(f"Using device: {device}")
        bundle = load_dataset_bundle(args["dataset"], metadata_path=args["metadata_path"])
        ablation_root = run_planA_ablation(
            args=args,
            bundle=bundle,
            device=device,
            repo_root=repo_root,
            results_root=results_root,
        )
        print(f"Plan A ablation completed: {ablation_root}")
        return

    experiment_tag = args["experiment_tag"] or args["run_name"] or f"{args['model_variant']}-{args['temporal_pooling']}"
    experiment_group = args["experiment_group"]
    if not experiment_group and args["experiment_family"] == "exploratory":
        experiment_group = f"{args['dataset']}-{args['model_variant']}-sweep"
    if not experiment_group and args["output_dir"]:
        experiment_group = Path(args["output_dir"]).name

    layout = initialize_layout(
        build_experiment_layout(
            results_root=results_root,
            family=args["experiment_family"],
            dataset=args["dataset"],
            model_variant=args["model_variant"],
            split_mode=args["split_mode"],
            experiment_tag=experiment_tag,
            experiment_group=experiment_group,
        )
    )
    run_dir = layout.run_dir

    print(f"Using device: {device}")
    print(f"Run dir: {run_dir}")

    bundle = load_dataset_bundle(args["dataset"], metadata_path=args["metadata_path"])
    dataset_summary = {
        "dataset": bundle.dataset_name,
        "num_samples": int(bundle.x_all.shape[0]),
        "num_classes": int(bundle.num_classes),
        "input_size": int(bundle.input_size),
        "class_names": bundle.class_names,
    }
    env_info = collect_environment_info(repo_root)
    write_json(layout.config_dir / "cli_args.json", to_serializable(vars(args_ns)))
    write_json(layout.config_dir / "resolved_config.json", to_serializable(args))
    write_json(layout.config_dir / "dataset_summary.json", dataset_summary)
    write_json(layout.config_dir / "env.json", env_info)

    manifest = build_run_manifest(
        layout=layout,
        cli_args=vars(args_ns),
        resolved_config=args,
        dataset_summary=dataset_summary,
        env=env_info,
        seed_list=args["seed_list"],
        group_key=args["group_key"],
        status="running",
    )
    write_manifest_and_registry(layout, manifest)
    write_run_readme(layout, manifest)

    seed_outputs = []
    for seed in args["seed_list"]:
        print(f"=== Seed {seed} ===")
        set_global_seed(int(seed), deterministic=True)
        out = train_one_seed(
            seed=int(seed),
            args=args,
            bundle=bundle,
            run_dir=run_dir,
            results_root=results_root,
            splits_dir=layout.splits_dir,
            device=device,
        )
        seed_outputs.append(out)

    aggregate_outputs(run_dir=run_dir, bundle=bundle, seed_outputs=seed_outputs, save_confusion_png=args["save_confusion_png"])
    manifest["status"] = "completed"
    write_manifest_and_registry(layout, manifest)
    write_run_readme(layout, manifest)
    print(f"Protocol run completed: {run_dir}")


if __name__ == "__main__":
    main()
