import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dual_domain_model import build_model
from protocol_data import build_dataloaders, generate_or_load_split, load_dataset_bundle
from protocol_utils import parse_int_list, write_csv, write_json
from run_protocol import SoftTargetCrossEntropy, evaluate, resolve_device, run_attention_diagnostics, str2bool


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str2bool(value)


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "config" / "resolved_config.json"
    if not path.exists():
        path = run_dir / "run_config_snapshot.json"
    if not path.exists():
        raise FileNotFoundError(f"missing run config: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg.setdefault("temporal_pooling", "attn")
    cfg.setdefault("freq_feature_dim", 64)
    cfg.setdefault("fusion_hidden_dim", 64)
    cfg.setdefault("freq_use_abs", False)
    cfg.setdefault("freq_eps", 1e-8)
    cfg.setdefault("split_mode", "predefined")
    cfg.setdefault("group_key", "auto")
    cfg.setdefault("train_ratio", 0.7)
    cfg.setdefault("val_ratio", 0.1)
    cfg.setdefault("test_ratio", 0.2)
    cfg.setdefault("stratified_random_split", True)
    cfg.setdefault("reuse_splits", True)
    cfg.setdefault("augment_train", False)
    cfg.setdefault("num_workers", 0)
    cfg.setdefault("metadata_path", None)
    cfg.setdefault("model_variant", "baseline")
    cfg.setdefault("attention_dim", 32)
    cfg.setdefault("hidden_size", 128)
    cfg.setdefault("batchsize", 128)
    cfg.setdefault("seed_list", [42])
    return cfg


def _resolve_seed_list(raw: Any) -> List[int]:
    if isinstance(raw, str):
        return parse_int_list(raw)
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw]
    raise ValueError("seed_list must be string or list")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run attention diagnostics on an existing protocol run directory.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to an experiment run directory")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--attention-max-export-samples", type=int, default=0)
    parser.add_argument("--attention-shuffle-eval", type=str, default="true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    cfg = _load_run_config(run_dir)
    if str(cfg.get("temporal_pooling", "attn")).strip().lower() != "attn":
        raise ValueError("this script only applies to attention pooling runs (temporal_pooling=attn)")

    device = resolve_device(args.device)
    dataset_name = str(cfg["dataset"])
    bundle = load_dataset_bundle(dataset_name, metadata_path=cfg.get("metadata_path"))
    criterion = SoftTargetCrossEntropy()

    seed_rows: List[Dict[str, Any]] = []
    attention_shuffle_eval = _as_bool(args.attention_shuffle_eval, True)
    max_export = int(args.attention_max_export_samples)

    for seed in _resolve_seed_list(cfg["seed_list"]):
        seed_dir = run_dir / "seeds" / f"seed_{seed}"
        if not seed_dir.exists():
            seed_dir = run_dir / f"seed_{seed}"
        metrics_path = seed_dir / "metrics.json"
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

        split_file = run_dir / "splits" / f"split_seed{seed}.json"
        if not split_file.exists():
            split_file = run_dir / "splits" / f"{bundle.dataset_name}_{cfg['split_mode']}_{cfg['group_key']}_seed{seed}.json"
        split, _ = generate_or_load_split(
            bundle=bundle,
            split_file=split_file,
            split_mode=str(cfg["split_mode"]),
            group_key=str(cfg["group_key"]),
            seed=int(seed),
            train_ratio=float(cfg["train_ratio"]),
            val_ratio=float(cfg["val_ratio"]),
            test_ratio=float(cfg["test_ratio"]),
            stratified_random=_as_bool(cfg["stratified_random_split"], True),
            reuse_existing=_as_bool(cfg["reuse_splits"], True),
        )

        _, _, test_loader, _ = build_dataloaders(
            bundle=bundle,
            split=split,
            train_batch_size=int(cfg["batchsize"]),
            eval_batch_size=int(cfg["batchsize"]),
            num_workers=int(cfg["num_workers"]),
            seed=int(seed),
            augment_train=_as_bool(cfg["augment_train"], False),
        )

        checkpoint_path = Path(
            metrics_payload.get(
                "checkpoint_path",
                str(seed_dir / "checkpoint_best.pt"),
            )
        )
        if not checkpoint_path.is_absolute():
            checkpoint_path = (repo_root / checkpoint_path).resolve()
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = build_model(
            model_variant=str(cfg["model_variant"]),
            input_dim=int(bundle.input_size),
            hidden_dim=int(cfg["hidden_size"]),
            attention_dim=int(cfg["attention_dim"]),
            output_dim=int(bundle.num_classes),
            freq_feature_dim=int(cfg["freq_feature_dim"]),
            fusion_hidden_dim=int(cfg["fusion_hidden_dim"]),
            freq_use_abs=_as_bool(cfg["freq_use_abs"], False),
            freq_eps=float(cfg["freq_eps"]),
            temporal_pooling=str(cfg["temporal_pooling"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)

        clean_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=bundle.num_classes,
            class_names=bundle.class_names,
            seed=int(seed),
        )
        diag = run_attention_diagnostics(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=bundle.num_classes,
            class_names=bundle.class_names,
            seed=int(seed),
            save_dir=seed_dir,
            max_export_samples=max_export,
            run_shuffle_eval=attention_shuffle_eval,
        )
        if diag is None:
            continue

        row: Dict[str, Any] = {
            "seed": int(seed),
            "accuracy": float(clean_metrics["accuracy"]),
            "macro_f1": float(clean_metrics["macro_f1"]),
            "normalized_entropy_mean": float(diag["normalized_entropy_mean"]),
            "normalized_entropy_std": float(diag["normalized_entropy_std"]),
            "mean_sample_l1_to_global_profile": float(diag["mean_sample_l1_to_global_profile"]),
            "top_timestep_unique_ratio": float(diag["top_timestep_unique_ratio"]),
        }
        shuffle_control = diag.get("shuffle_control")
        if isinstance(shuffle_control, dict):
            row["shuffle_accuracy"] = float(shuffle_control["accuracy"])
            row["shuffle_macro_f1"] = float(shuffle_control["macro_f1"])
            row["delta_accuracy_shuffle"] = float(row["accuracy"] - row["shuffle_accuracy"])
            row["delta_macro_f1_shuffle"] = float(row["macro_f1"] - row["shuffle_macro_f1"])

        metrics_payload.update(
            {
                "attn_entropy_norm_mean": row["normalized_entropy_mean"],
                "attn_entropy_norm_std": row["normalized_entropy_std"],
                "attn_top_timestep_unique_ratio": row["top_timestep_unique_ratio"],
                "attn_l1_to_mean_profile": row["mean_sample_l1_to_global_profile"],
            }
        )
        if "shuffle_accuracy" in row:
            metrics_payload.update(
                {
                    "shuffle_accuracy": row["shuffle_accuracy"],
                    "shuffle_macro_f1": row["shuffle_macro_f1"],
                    "delta_accuracy_shuffle": row["delta_accuracy_shuffle"],
                    "delta_macro_f1_shuffle": row["delta_macro_f1_shuffle"],
                }
            )
        write_json(metrics_path, metrics_payload)
        write_csv(seed_dir / "metrics.csv", [metrics_payload], fieldnames=list(metrics_payload.keys()))

        seed_rows.append(row)

    if not seed_rows:
        raise RuntimeError("no attention diagnostics were generated")

    aggregate_dir = run_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    write_csv(aggregate_dir / "attention_diagnostics_summary.csv", seed_rows, fieldnames=list(seed_rows[0].keys()))

    numeric_keys = [
        key
        for key in seed_rows[0].keys()
        if key != "seed" and all(isinstance(row.get(key), (int, float)) for row in seed_rows if key in row)
    ]
    summary_rows = []
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in seed_rows], dtype=np.float64)
        summary_rows.append({"metric": key, "mean": float(values.mean()), "std": float(values.std(ddof=0))})
    write_csv(aggregate_dir / "attention_diagnostics_mean_std.csv", summary_rows, fieldnames=["metric", "mean", "std"])

    print(f"attention diagnostics completed: {run_dir}")


if __name__ == "__main__":
    main()
